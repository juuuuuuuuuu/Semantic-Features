import numpy as np
import pandas as pd
import os
import cv2
import random
from scipy.linalg import expm
import pykitti
import scipy.spatial.transform.rotation as r
from tools import utils
# World is twisted so we need to transform.


dt = 1/20
max_vis = 30.0
std_w = 0
std_v = 0
# fraction of particles to project onto road
alpha = 0.2
# probability of detection of a point
rho = 0.6
# P0 is a design parameter specifying the probability that a visible map point is occluded
P0 = 0.2
# design PMF pr of detection d under not occluded and the map detections D
# pr_d = vector of probabilities for each class find via validation class mismatches? detector property?
w = np.array([0.0, 0.0, 1.0])
v_t = np.array([1.0, 0.0, 0.0])
ROOT_DIR = os.path.abspath('')
results_path = os.path.join(ROOT_DIR, "results/_results.txt")
instances_path = os.path.join(ROOT_DIR, "content/kitti_dataset/dataset/sequences/08/instances_2")
CNN_PMF_PATH = os.path.join(ROOT_DIR, 'results/PMFs')
basedir = 'content/kitti_dataset/dataset'
sequence = '08'
CNN_PMF_PATH = os.path.join(ROOT_DIR, 'results/PMFs')
cnn_pmf = np.load(CNN_PMF_PATH + '.npy') # one row is the pmf of the cnn detecting the class, when the gt corresponding to the row is present

R_init = lambda th: np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])


w_x = lambda w: np.array([[0.0, -w[2], w[1]],
                          [w[2], 0.0, -w[0]],
                          [-w[1], w[0], 0.0]])


def pr_delt(x, U):
    dist = L2_norm(x, U)
    if dist > max_dist:
        return 0.0
    else:
        return rho*(1-P0)


def L2_norm(x, y):
    return ((x-y)**2).sum()**0.5


e_w = lambda w, th: np.identity(3) + w_x(w)*np.sin(th) + w_x(w).dot(w_x(w))*(1-np.cos(th))
#transl = np.identity(3) - e_w.dot(np.cross(w,v)) + w.dot(w.T).dot(v)*th

transl = lambda v_t, q_v: dt*v_t + q_v

twist = np.array([e_w])
#print(e_w(w,2))
#print(transl(v_t, [0.1, 0.1, 0.1]))


def load_frame(path):
    results = pd.read_csv(path, header=None, sep=' ')
    data = np.array(results.values)
    U = []
    D = []
    image_ids = []
    for row in data:
        landm_path = row[3]
        pcl = np.load(landm_path + '.npy')
        U.append(pcl)
        D.append(row[2])
        image_ids.append(row[0])
    image_ids = np.unique(image_ids)
    return U, D, image_ids

def weighted_choice(objects, weights):
    """ returns randomly an element from the sequence of 'objects',
        the likelihood of the objects is weighted according
        to the sequence of 'weights', i.e. percentages."""

    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]


def get_instances(path, image_id):
    image_path = os.path.join(path, 'L{}.png'.format(image_id))
    mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return mask_image


def motion_update(w_t, v_t):
    q_w = np.random.normal(0.0, std_w, size=1)
    q_w = np.array([0.0, 0.0, q_w])
    q_v = np.random.normal(0.0, std_v, size=2)
    q_v = np.array([q_v[0], q_v[1], 0.0])
    twist = np.zeros((4, 4))
    twist[0:3, 0:3] = expm(w_x(w_t + q_w))
    twist[0:3, 3] = v_t + q_v
    twist[3, 3] = 1.0
    return twist


def proj_trajectory(sorted_indices, poses):
    p1 = poses[sorted_indices[0]][0:3, 3]
    p2 = poses[sorted_indices[1]][0:3, 3]
    s = p2 - p1
    proj_len = (position - p1).dot(s)
    if proj_len < 0:
        proj = p1
    elif proj_len > np.linalg.norm(s):
        proj = p2
    else:
        proj = proj_len / (s.dot(s)) * s + p1
    return proj


def get_delta_rot(rot_1, rot_2):
    return r.Rotation.from_matrix(np.dot(rot_1.T, rot_2)).as_rotvec()


def get_gt_velocities(poses):
    linear_velocities = []
    angular_velocities = []
    for i in range(len(poses) - 1):
        linear_v = poses[i][:3, :3].T.dot(poses[i + 1][:3, 3] - poses[i][:3, 3])
        #linear_v = (poses[i + 1][:3, 3] - poses[i][:3, 3])
        linear_velocities.append(linear_v)
        #rotational_v = poses[i][:3, :3].T.dot(get_delta_rot(poses[i + 1][:3, :3], poses[i][:3, :3]))
        rotational_v = (get_delta_rot(poses[i + 1][:3, :3], poses[i][:3, :3]))
        angular_velocities.append(rotational_v)

    return linear_velocities, angular_velocities

def measurement_update(image_id, particles, U, D):
    """ takes an image_id as measurment, as well as the particles, and 3D-map (U, D)
    """
    # select local map to project into image plane
    instances_path = os.path.join(ROOT_DIR, "content/kitti_dataset/dataset/sequences/08/instances_2")
    image_ids = os.listdir(os.path.join(instances_path))
    instance_im = cv2.imread(os.path.join(instances_path, '{}'.format(image_ids[image_id])))
    print(os.path.join(instances_path, '{}'.format(image_ids[image_id])))
    weights = np.zeros(particles.shape)
    for i in range(particles.size):
        map = []
        feat = []
        for j, u in enumerate(U):
            if u[0:3, :, 0].size > 0:
                coord_mean = u[0:3, :, 0].mean(axis=1)
                # print(coord_mean.shape)
                dist = np.linalg.norm(coord_mean - particles[i][0:3, 3])
                if dist < max_vis:
                    map.append(u[0:3, :, 0].T)
                    feat.append(D[j])

        depth_image, label_image = utils.pcls_to_image_labels(map, feat, particles[i], Kmat, (370, 1226))
        im_prob = 1
        for u in range(1226):
            for v in range(370):
                if label_image[u, v]:
                    print(cnn_pmf)
                    print(label_image)
                    print(instance_im)
                    im_prob = im_prob * 0.8 * cnn_pmf[label_image[u, v], instance_im[u, v]]
                    print(im_prob)
        weights[i] = im_prob
    return weights
if __name__ == '__main__':


    dataset = pykitti.odometry(basedir, sequence)
    Kmat = dataset.calib.P_rect_20
    T_w0_w = np.array([[0., 0., 1., 0.],
                       [-1., 0., 0., 0.],
                       [0., -1., 0., 0.],
                       [0., 0., 0., 1.]])
    T_cam0_cam2 = np.linalg.inv(dataset.calib.T_cam0_velo).dot(dataset.calib.T_cam2_velo)
    #Kmat = np.concatenate((Kmat, np.array([0, 0, 0, 1]).reshape(4,1)), axis=1)
    print(Kmat)
    cnn_pmf = np.load(CNN_PMF_PATH + '.npy') # one row is the pmf of the cnn detecting the class, when the gt corresponding to the row is present
    U, D, image_ids = load_frame(results_path)

    # initialize particles and weights
    N = 100
    poses = []
    #for image_id in image_ids:
     #   poses.append((dataset.poses[image_id].dot(T_cam0_cam2)))


    poses = [T_w0_w.dot(dataset.poses[image_id].dot(T_cam0_cam2)) for image_id in image_ids]
    v, w = get_gt_velocities(poses)
    particles = np.zeros((N, 4, 4))
    weights = np.ones(N)*1/N
    for i in range(N):
        # th = np.pi * np.random.uniform(0, 2)
        # particles[i,0:3, 0:3] = R_init(th)
        # particles[i, 3,:] = [0.0, 0.0, 0.0, 1]
        # randpose = np.random.randint(0,len(poses))
        # particles[i, 0:3, 3] = poses[randpose][0:3,3]
        particles[i] = poses[0]
    print(particles[0])

    particle_poses_all = []

    # loop trough all measurements
    for time, image_id in enumerate(image_ids):
        #pose = T_w0_w.dot(dataset.poses[image_id].dot(T_cam0_cam2))

        # measure w_t and v_t
        w_t, v_t = w[time], v[time]
        # Motion update
        particle_poses = np.zeros((N, 3))
        for i in range(N):
            particles[i] = motion_update(w_t, v_t).dot(particles[i])
            particle_poses[i] = particles[i][0:3, 3]
        particle_poses_all.append(particle_poses)
        measurement_model_path = os.path.join(ROOT_DIR, "particle_poses")

        # project particles onto trajectory
        proj_ind = np.random.choice(list(range(N)), int(N * alpha))
        for i in proj_ind:
            position = particles[i][0:3, 3]
            dist = ((np.asarray(poses)[:, 0:3, 3] - position) ** 2).sum(axis=1) ** 0.5
            sorted_indices = np.argpartition(dist, 1)
            proj = proj_trajectory(sorted_indices, poses)
            # print("proj: {}".format(proj))
            # print("position: {}".format(position))
            particles[i][0:3, 3] = proj

        # calculate weights with measurement update
        print(image_id)
        weights = measurement_update(image_id, particles, U, D)
        # resample
        for i in range(N):
            particles[i] = weighted_choice(particles, weights)


    np.save(measurement_model_path, particle_poses_all, allow_pickle=False)
    print(particles[0])
