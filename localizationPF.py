import numpy as np
import pandas as pd
import os
import cv2
from random import random
from scipy.linalg import expm
import pykitti
import scipy.spatial.transform.rotation as r
from tools import utils
import velocity_measurement
import json
from operator import itemgetter

from matplotlib import pyplot as plt
# World is twisted so we need to transform.


R_init = lambda th: np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])


w_x = lambda w: np.array([[0.0, -w[2], w[1]],
                          [w[2], 0.0, -w[0]],
                          [-w[1], w[0], 0.0]])


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

def get_delta_rot(rot_1, rot_2):
    return r.Rotation.from_matrix(np.dot(rot_1.T, rot_2)).as_rotvec()


def get_gt_velocities(poses):
    linear_velocities = []
    angular_velocities = []
    for i in range(len(poses) - 1):
        #linear_v = poses[i][:3, :3].T.dot(poses[i + 1][:3, 3] - poses[i][:3, 3])
        linear_v = (poses[i + 1][:3, 3] - poses[i][:3, 3])
        linear_velocities.append(linear_v)
        #rotational_v = poses[i][:3, :3].T.dot(get_delta_rot(poses[i + 1][:3, :3], poses[i][:3, :3]))
        rotational_v = (get_delta_rot(poses[i + 1][:3, :3], poses[i][:3, :3]))
        angular_velocities.append(rotational_v)

    return linear_velocities, angular_velocities

def L2_norm(x, y):
    return ((x - y)**2).sum()**0.5

class Particle_Filter():
    def __init__(self, N, std_w, std_v):
        self.N = N
        self.dt = 1/20
        self.max_vis = 30.0
        self.std_w = std_w
        self.std_v = std_v
        # fraction of particles to project onto road
        self.alpha = 0.2
        # probability of detection of a point
        self.rho = 0.6
        # P0 is a design parameter specifying the probability that a visible map point is occluded
        self.P0 = 0.2
# design PMF pr of detection d under not occluded and the map detections D
# pr_d = vector of probabilities for each class find via validation class mismatches? detector property?

        self.ROOT_DIR = os.path.abspath('')
        self.results_path = os.path.join(self.ROOT_DIR, "results/_results.txt")
        self.instances_path = os.path.join(self.ROOT_DIR, "content/kitti_dataset/dataset/sequences/08/instances_2")
        self.CNN_PMF_PATH = os.path.join(self.ROOT_DIR, 'results/PMFs')
        self.CLASS_MARGINAL = os.path.join(self.ROOT_DIR, 'detection_probabilities')
        self.class_marginal = np.load(self.CLASS_MARGINAL + '.npy')
        self.basedir = 'content/kitti_dataset/dataset'
        self.sequence = '08'
        self.CNN_PMF_PATH = os.path.join(self.ROOT_DIR, 'results/PMFs')
        self.cnn_pmf = np.load(self.CNN_PMF_PATH + '.npy') # one row is the pmf of the cnn detecting the class, when the gt corresponding to the row is present
        self.dataset = pykitti.odometry(self.basedir, self.sequence)
        self.Kmat = self.dataset.calib.P_rect_20
        self.T_w0_w = np.array([[0., 0., 1., 0.],
                           [-1., 0., 0., 0.],
                           [0., -1., 0., 0.],
                           [0., 0., 0., 1.]])
        self.T_cam0_cam2 = np.linalg.inv(self.dataset.calib.T_cam0_velo).dot(self.dataset.calib.T_cam2_velo)
        with open("results.json") as json_file:
            all_data = json.load(json_file)

            # sorting alldata for image_id
        all_data = all_data['results']
        self.all_data_sort = []
        for x in sorted(all_data, key=itemgetter('image_id')):
            self.all_data_sort.append(x)

        print(self.class_marginal)
    def pr_delt(self, x, U):
        dist = L2_norm(x, U)
        if dist > self.max_dist:
            return 0.0
        else:
            return self.rho * (1 - self.P0)


    def load_frame(self, indices):
        results = pd.read_csv(self.results_path, header=None, sep=' ')
        data = np.array(results.values)
        U = []
        D = []
        image_ids = []
        for row in data:
            if row[0] in indices:
                landm_path = row[3]
                pcl = np.load(landm_path + '.npy')
                U.append(pcl)
                D.append(row[2])
                image_ids.append(row[0])
        image_ids = np.unique(image_ids)
        return U, D, image_ids


    def motion_update(self, w_t, v_t):
        q_w = np.random.normal(0.0, self.std_w, size=1)
        q_w = np.array([0.0, 0.0, q_w])
        q_v = np.random.normal(0.0, self.std_v, size=2)
        q_v = np.array([q_v[0], q_v[1], 0.0])
        twist = np.zeros((4, 4))
        twist[0:3, 0:3] = expm(w_x(w_t + q_w))
        twist[0:3, 3] = v_t + q_v
        twist[3, 3] = 1.0
        return twist


    def proj_trajectory(self, sorted_indices, poses, position):
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


    def measurement_update(self, image_id, particles, U, D):
        """ takes an image_id as measurement, as well as the particles, and 3D-map (U, D)
        """
        # select local map to project into image plane
        instances_to_classes = [0] + self.all_data_sort[image_id]['classes']
        def instances_to_classes_map(x):
            return instances_to_classes[int(x)]
        cnn_pmf_map = lambda x: self.cnn_pmf[x]
        class_marginal_map = lambda x: self.class_marginal[x]
        image_id = "L{:06d}.png".format(image_id)

        print(image_id)
        instance_im = cv2.imread(os.path.join(self.instances_path, '{}'.format(image_id)), cv2.IMREAD_GRAYSCALE)
        weights = np.zeros(self.N)
        for i in range(self.N):
            local_map = []
            feat = []
            for j, u in enumerate(U):
                if u[0:3, :, 0].size > 0:
                    coord_mean = u[0:3, :, 0].mean(axis=1)
                    # print(coord_mean.shape)
                    dist = np.linalg.norm(coord_mean - particles[i][0:3, 3])
                    if dist < self.max_vis:
                        local_map.append(u[0:3, :, 0].T)
                        feat.append(D[j])

            depth_image, label_image = utils.pcls_to_image_labels(local_map, feat, particles[i], self.Kmat, (370, 1226))
            im_prob = 1

            #get a boolean mask of all pixels that are matched with the map
            proj_mask = np.where(~np.isnan(label_image), True, False)
            map_classes = list(map(int, label_image[proj_mask]))
            pred_image = instance_im[proj_mask]
            pred_image = np.array(list(map(instances_to_classes_map, pred_image)))
            detect_prob = np.array(list(map(cnn_pmf_map, zip(map_classes, pred_image)))) * 0.9 + 0.1
            im_prob = np.cumsum(detect_prob)[-1]
            im_prob = im_prob / np.cumsum(np.array(list(map(class_marginal_map, pred_image))) * 0.1)[-1]
            # for u in range(1226):
            #     for v in range(370):
            #         if not np.isnan(label_image[v, u]):
            #             #print(self.cnn_pmf[int(label_image[v, u]), instance_im[v, u]])
            #             if instance_im[v, u]:
            #                 pred_label = instances_to_classes[instance_im[v, u]]
            #             else:
            #                 pred_label = 0
            #             print("class marginal: {}, label: {}".format(self.class_marginal[pred_label], pred_label))
            #             im_prob = im_prob * (self.cnn_pmf[int(label_image[v, u]), pred_label] * 0.9 + im_prob * 0.1)/self.class_marginal[pred_label]*0.1
            #             print(im_prob)

            weights[i] = im_prob
        return weights

    def run(self, mapping_indices, localization_indices):

        # Kmat = np.concatenate((Kmat, np.array([0, 0, 0, 1]).reshape(4,1)), axis=1)

        U, D, image_ids = self.load_frame(mapping_indices)

        # initialize particles and weights
        N = self.N

        mapping_poses = [self.T_w0_w.dot(self.dataset.poses[image_id].dot(self.T_cam0_cam2)) for image_id in mapping_indices]

        particles = np.zeros((N, 4, 4))
        weights = np.ones(N) * 1 / N
        for i in range(N):
            # th = np.pi * np.random.uniform(0, 2)
            # particles[i,0:3, 0:3] = R_init(th)
            # particles[i, 3,:] = [0.0, 0.0, 0.0, 1]
            # randpose = np.random.randint(0,len(poses))
            # particles[i, 0:3, 3] = poses[randpose][0:3,3]
            particles[i] = mapping_poses[0]

        particle_poses_all = []
        measurement_model_path = os.path.join(self.ROOT_DIR, "particle_poses")
        # loop trough all measurements
        loc_pose_ind = localization_indices + [localization_indices[-1]+1]
        localization_poses = [self.T_w0_w.dot(self.dataset.poses[image_id].dot(self.T_cam0_cam2)) for image_id in loc_pose_ind]
        # v, w = get_gt_velocities(localization_poses)
        v, w = velocity_measurement.get_gt_velocities_vehicle(localization_poses)
        for time, image_id in enumerate(localization_indices):
            # pose = T_w0_w.dot(dataset.poses[image_id].dot(T_cam0_cam2))

            # measure w_t and v_t
            w_t, v_t = w[time], v[time]
            # Motion update
            particle_poses = np.zeros((N, 3))
            for i in range(N):
                # Get noise vectors.
                q_w = np.random.normal(0.0, self.std_w, size=1)
                q_w = np.array([0.0, q_w, 0.0])
                q_v = np.random.normal(0.0, self.std_v, size=2)
                q_v = np.array([q_v[0], 0.0, q_v[1]])
                # particles[i] = self.motion_update(w_t, v_t).dot(particles[i])
                particles[i] = velocity_measurement.process_particle(particles[i], v_t, w_t, q_v, q_w)
                particle_poses[i] = particles[i][0:3, 3]
            particle_poses_all.append(particle_poses)

            # project particles onto trajectory
            proj_ind = np.random.choice(list(range(N)), int(N * self.alpha))
            for i in proj_ind:
                particle_position = particles[i][0:3, 3]
                dist = ((np.asarray(mapping_poses)[:, 0:3, 3] - particle_position) ** 2).sum(axis=1) ** 0.5
                sorted_indices = np.argpartition(dist, 1)
                proj = self.proj_trajectory(sorted_indices, mapping_poses, particle_position)
                # print("proj: {}".format(proj))
                # print("particle_position: {}".format(particle_position))
                particles[i][0:3, 3] = proj

            # calculate weights with measurement update
            weights = self.measurement_update(image_id, particles, U, D)
            # normalize
            sum_weights = weights.sum()
            weights = weights/sum_weights
            # resample
            particle_ind = list(range(N))
            particle_ind = np.random.choice(particle_ind, p=weights)
            new_particles = particles
            for i in range(N):
                new_particles[i] = particles[i]

        np.save(measurement_model_path, particle_poses_all, allow_pickle=False)
if __name__ == '__main__':
    filter = Particle_Filter(10, std_w=0.02, std_v=0.2)
    mapping_indices = list(range(20))
    localization_indices = list(range(20))
    filter.run(mapping_indices, localization_indices)


