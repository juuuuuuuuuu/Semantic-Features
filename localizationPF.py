import numpy as np
import pandas as pd
import os
import cv2
from scipy.linalg import expm


dt = 1/20
max_vis = 30.0
std_w = 0.01
std_v = 3
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
    poses = []
    image_id = data[0][0]
    for row in data:
        landm_path = row[3]
        pcl = np.load(landm_path + '.npy')
        U.append(pcl)
        D.append(row[2])
        if row[0] != image_id:
            pose_path = row[8]
            pose = np.load(pose_path + '.npy')
            poses.append(pose)
            image_id = row[0]
    return U, D, poses

def get_instances(path, image_id):
    image_path = os.path.join(path, 'L{}.png'.format(image_id))
    mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return mask_image

def motion_update(w_t, v_t):
    q_w = np.random.normal(0.0, std_w, size=3)
    q_v = np.random.normal(0.0, std_v, size=3)
    twist = np.zeros((4,4))
    twist[0:3,0:3] = expm(w_x(w_t*dt + q_w))
    twist[0:3,3] = dt*v_t +q_v
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

if __name__ == '__main__':
    ROOT_DIR = os.path.abspath('')
    results_path = os.path.join(ROOT_DIR, "results/_results.txt")
    instanes_path = os.path.join(ROOT_DIR, "content/kitti_dataset/dataset/sequences/08/instances_2")

    U, D, poses = load_frame(results_path)
    # initialize particles and weights
    N = 100
    particles = np.zeros((N, 4, 4))
    weights = np.ones(N)*1/N
    for i in range(N):
        th = np.pi * np.random.uniform(0, 2)
        particles[i,0:3, 0:3] = R_init(th)
        particles[i, 3,:] = [0.0, 0.0, 0.0, 1]
        randpose = np.random.randint(0,len(poses))
        particles[i, 0:3, 3] = poses[randpose][0:3,3]
    print(particles[0])

    # Motion update

    #measure w_t and v_t
    w_t = np.random.normal(0.0, std_w, size=3)
    v_t = np.random.normal(0.0, std_v, size=3)
    for i in range(N):
        particles[i] = motion_update(w_t, v_t).dot(particles[i])
    print(particles[0])

    # project particles onto trajectory
    proj_ind = np.random.choice(list(range(N)),int(N*alpha))
    for i in proj_ind:
        position = particles[i][0:3,3]
        print(np.asarray(poses)[:, 0:3, 3])
        dist = ((np.asarray(poses)[:,0:3,3] - position)**2).sum(axis=1)**0.5
        sorted_indices = np.argpartition(dist,1)
        proj = proj_trajectory(sorted_indices, poses)
        print("proj: {}".format(proj))
        print("position: {}".format(position))
        particles[i][0:3,3] = proj

    # select local map to project into image plane
    for i in range(N):
        map = []
        for u in U:
            coord_mean = u[0:3,:,0].mean(axis=1)
            print(coord_mean.shape)
            dist = np.linalg.norm(coord_mean - particles[i][0:3,3])
            if dist > max_vis:
                map.append(u[0:3,:,0])

        #project map points into image plane
        R_t = particles[i][0:3,0:3].T
        t = particles[i][0:3,3]
        np.concatenate(R_t, -R_t.dot(t),axis=1)