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
import optical_flow
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
    def __init__(self, N, std_w, std_v, gamma, bias_w_std):
        self.N = N
        self.dt = 1/20
        self.max_vis = 40.0
        self.std_w = std_w
        self.std_v = std_v
        self.bias_w_std = bias_w_std
        self.gamma = gamma
        # fraction of particles to project onto road
        self.alpha = 0.0
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

        self.rescale_factor = 5
        self.prob_mask = np.zeros((int(370 / self.rescale_factor), int(1226 / self.rescale_factor), 17))
        self.previous_image = None

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


    def process_particle(self, pose, rel_v, rel_w, std_v, std_w, gamma, bias_w_std, bias_old):
        # Get noise vectors.

        q_v = np.random.normal(0.0, std_v, size=2)
        q_v = np.array([q_v[0], 0.0, q_v[1]])

        bias_w = (1-gamma)*bias_old + np.random.normal(0., bias_w_std, 1)
        q_w = np.random.normal(bias_w, std_w, size=1)
        q_w = np.array([0.0, q_w, 0.0])
        
        v_world = np.squeeze(pose[:3, :3].dot((rel_v + q_v).reshape(3, 1)))
        w_ = rel_w + q_w
        old_pos = pose[:3, 3]
        old_r = pose[:3, :3]
        new_pos = old_pos + v_world
        new_r = old_r.dot(r.Rotation.from_rotvec(w_).as_matrix())
        out_pose = np.eye(4)
        out_pose[:3, :3] = new_r
        out_pose[:3, 3] = new_pos
        return out_pose, bias_w


    def measurement_update(self, image_id, particles, U, D):
        """ takes an image_id as measurement, as well as the particles, and 3D-map (U, D)
        """
        # select local map to project into image plane
        instances_to_classes = [0] + self.all_data_sort[image_id]['classes']
        def instances_to_classes_map(x):
            return instances_to_classes[int(x)]
        cnn_pmf_map = lambda x: self.cnn_pmf[x]
        class_marginal_map = lambda x: self.class_marginal[x]
        image_id_ = image_id
        image_id = "L{:06d}.png".format(image_id)

        print(image_id)
        instance_im = cv2.imread(os.path.join(self.instances_path, '{}'.format(image_id)), cv2.IMREAD_GRAYSCALE)
        rescale_factor = 5
        instance_im = cv2.resize(instance_im, (int(instance_im.shape[1] / rescale_factor),
                                               int(instance_im.shape[0] / rescale_factor)),
                                 cv2.INTER_NEAREST)

        classes_im = np.vectorize(instances_to_classes_map)(instance_im)

        next_image = cv2.imread("content/kitti_dataset/dataset/sequences/08/image_2/{:06d}.png"
                                .format(image_id_), cv2.IMREAD_UNCHANGED)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
        if self.previous_image is None:
            self.previous_image = next_image

        flow = optical_flow.calculate_flow(self.previous_image, next_image)
        flow = cv2.resize(flow, (int(flow.shape[1] / self.rescale_factor),
                                 int(flow.shape[0] / self.rescale_factor)),
                          cv2.INTER_LINEAR) / self.rescale_factor
        self.previous_image = next_image
        self.prob_mask = optical_flow.update_mask(self.prob_mask, classes_im, flow, 0.6)

        weights = np.zeros(self.N)
        num_projections = []
        pred_images = []
        proj_masks = []
        for i in range(self.N):
            local_map = []
            feat = []
            means = []
            for j, u in enumerate(U):
                if u[0:3, :, 0].size > 0:
                    coord_mean = u[0:3, :, 0].mean(axis=1)
                    # print(coord_mean.shape)
                    dist = np.linalg.norm(coord_mean - particles[i][0:3, 3])
                    if dist < self.max_vis:
                        local_map.append(u[0:3, :, 0].T)
                        means.append(coord_mean)
                        feat.append(D[j])
            intrinsics = self.Kmat.copy()
            # print(self.Kmat)
            intrinsics[:2, :3] = intrinsics[:2, :3] / rescale_factor
            # print(intrinsics)
            depth_image, label_image = utils.pcls_to_image_labels(local_map, means, feat, particles[i], intrinsics,
                                                                  instance_im.shape)
            pred_images.append(label_image)
            # label_image, _ = utils.pcls_to_image_labels_with_occlusion(local_map, means, feat, particles[i], self.Kmat,
            #                                                            (instance_im.shape[0], instance_im.shape[1]),
            #                                                            0.2, 100.)
            im_prob = 1

            if True:
                # get a boolean mask of all pixels that are matched with the map
                proj_mask = np.where(~np.isnan(label_image), True, False)
                map_classes = list(map(int, label_image[proj_mask]))
                pred_image = instance_im[proj_mask]
                pred_image = np.array(list(map(instances_to_classes_map, pred_image)))
                num_px = pred_image.shape[0]
                # num_projections.append(num_px)
                num_projections.append(np.sum(np.logical_and(proj_mask, instance_im != 0)))
                proj_masks.append(np.logical_and(proj_mask, instance_im != 0))
                # print("Num landmarks in image: {}, number of map projections: {}".format(len(local_map), num_px))
                # num_px = min(1000, num_px)
                s = 2.
                if num_px > 0:
                    # (np.array(list(map(cnn_pmf_map, zip(map_classes, pred_image)))) * 0.9 + 0.1) ** (s / num_px)
                    detect_prob = ((np.array(list(map(cnn_pmf_map, zip(map_classes, pred_image)))) * 0.9 + 0.1) /
                                   np.array(list(map(class_marginal_map, pred_image)))) ** (s / num_px)
                    cumm = np.cumprod(detect_prob)
                    im_prob = cumm[-1]
                    # im_prob / np.cumprod((np.array(list(map(class_marginal_map, pred_image))) * 0.1) **
                    # im_prob = im_prob / np.cumprod((np.array(list(map(class_marginal_map, pred_image))) *
                    #                                 np.array(list(map(class_marginal_map, map_classes)))) **
                    #                                (s / num_px * 2.))[-1]
                else:
                    im_prob = 1e-300
                # im_prob = max(1e-300, num_px)
                # print("Num projections {}, probability {}".format(num_projections[-1], im_prob))
            else:
                seg_mask = np.where(classes_im != 0, True, False)
                map_image = np.where(np.isnan(label_image), 0., label_image)
                map_classes = list(map(int, map_image[seg_mask]))
                seg_classes = classes_im[seg_mask]
                num_px = seg_classes.shape[0]

                num_projections.append(np.sum(np.logical_and(seg_mask, ~np.isnan(label_image))))
                # proj_masks.append(np.logical_and(seg_mask, ~np.isnan(label_image)))
                proj_masks.append(seg_mask)

                s = 1.

                if num_px > 0:
                    detect_prob = (np.array(list(map(cnn_pmf_map, zip(seg_classes, map_classes)))) /
                                   np.array(list(map(class_marginal_map, map_classes)))) ** (s / num_px)
                    im_prob = np.cumprod(detect_prob)[-1]
                else:
                    im_prob = 1e-300

                # print("Num projections {}, probability {}".format(num_projections[-1], im_prob))

            weights[i] = im_prob

        max_weight = np.argmax(weights)
        max_image = pred_images[int(max_weight)]
        # max_image = pred_images[int(np.argmax(num_projections))]
        max_proj = proj_masks[int(max_weight)]
        # max_image = pred_images[0]
        # Take a screenshot of the particle vision.
        print("Max likely number of map projections: {}".format(num_projections[int(max_weight)]))
        print("Most projections: {}".format(max(num_projections)))
        if False:
            import imageio
            # f, ax = plt.subplots(2, 1, figsize=(15, 5))
            figure_for_jo = np.zeros((max_image.shape[0], max_image.shape[1], 3))
            #figure_for_jo = max_imgae
                # np.array(imageio.imread(
                # "content/kitti_dataset/dataset/sequences/08/image_2/{:06d}.png".format(image_id_))) / 255. / 3.
            print(figure_for_jo.shape)
            figure_for_jo_2 = np.zeros((max_image.shape[0], max_image.shape[1], 3))
            colors = np.zeros((17, 3))
            for k in range(17):
                np.random.seed(k)
                rgb = np.random.randint(255, size=(1, 3)) / 255.0
                colors[k, :] = rgb

            for k in range(max_image.shape[0]):
                for j in range(max_image.shape[1]):
                    label = max_image[k, j]
                    if not np.isnan(label):
                        label = int(label)
                        figure_for_jo[k, j, :] = colors[label, :]

            for k in range(instance_im.shape[0]):
                for j in range(instance_im.shape[1]):
                    label = instance_im[k, j]
                    if not np.isnan(label) and not label == 0:
                        label = int(label)
                        figure_for_jo_2[k, j, :] = colors[instances_to_classes_map(label), :]

            plt.close()
            f, ax = plt.subplots(2, 1, figsize=(15, 5))
            ax[0].imshow(figure_for_jo)
            # plt.pause(0.001)
            # plt.show(block=False)
            ax[1].imshow(max_proj)
            # ax[1].imshow(optical_flow.visualize_mask(self.prob_mask))
            plt.pause(0.001)
            # plt.show(block=False)
            # imageio.imwrite("figure_for_jo.png", figure_for_jo)
            # imageio.imwrite("figure_for_jo_2.png", figure_for_jo_2)
            # exit()

        return weights

    def run(self, mapping_indices, localization_indices):
        plt.ion()
        plt.show()

        # Kmat = np.concatenate((Kmat, np.array([0, 0, 0, 1]).reshape(4,1)), axis=1)

        U, D, image_ids = self.load_frame(mapping_indices)

        # initialize particles and weights
        N = self.N

        mapping_poses = [self.T_w0_w.dot(self.dataset.poses[image_id].dot(self.T_cam0_cam2)) for image_id in mapping_indices]

        particles = np.zeros((N, 4, 4))
        particles_wo_mm = np.zeros((N, 4, 4))
        bias_w_old = np.zeros((N, 1))
        bias_w_old_wo_mm = np.zeros((N, 1))
        weights = np.ones(N) * 1 / N
        # for i in range(N):
            # th = np.pi * np.random.uniform(0, 2)
            # particles[i,0:3, 0:3] = R_init(th)
            # particles[i, 3,:] = [0.0, 0.0, 0.0, 1]
            # randpose = np.random.randint(0,len(poses))
            # particles[i, 0:3, 3] = poses[randpose][0:3,3]
            # particles[i] = mapping_poses[0]

        particle_poses_all = []
        particle_poses_all_wo_mm = []
        measurement_model_path = os.path.join(self.ROOT_DIR, "particle_poses")
        # loop trough all measurements
        loc_pose_ind = localization_indices 
        localization_poses = [self.T_w0_w.dot(self.dataset.poses[image_id].dot(self.T_cam0_cam2)) for image_id in loc_pose_ind]
        np.save('gt_poses.npy', np.array([p[:3, 3] for p in localization_poses]), allow_pickle=False)
        for i in range(N):
            particles[i] = localization_poses[0]
            particles_wo_mm[i] = localization_poses[0]
        v, w = velocity_measurement.get_gt_velocities_vehicle(localization_poses, std_v=self.std_v, std_w=self.std_w, gamma=self.gamma, bias_w_std=self.bias_w_std)
        for time, image_id in enumerate(localization_indices):
            # pose = T_w0_w.dot(dataset.poses[image_id].dot(T_cam0_cam2))

            # measure w_t and v_t
            w_t, v_t = w[time], v[time]
            # Motion update
            particle_poses = np.zeros((N, 3))
            particle_poses_wo_mm = np.zeros((N, 3))
            for i in range(N):

                particles[i], bias_w_old[i] = self.process_particle(particles[i], v_t, w_t, std_v=self.std_v, std_w=self.std_w, gamma=self.gamma, bias_w_std=self.bias_w_std, bias_old=bias_w_old[i])
                particles_wo_mm[i], bias_w_old_wo_mm[i] = self.process_particle(particles_wo_mm[i], v_t, w_t, std_v=self.std_v, std_w=self.std_w, gamma=self.gamma, bias_w_std=self.bias_w_std, bias_old=bias_w_old_wo_mm[i])
                particle_poses[i] = particles[i][0:3, 3]
                particle_poses_wo_mm[i] = particles_wo_mm[i][0:3, 3]

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
            #self.measurement_update(image_id, particles, U, D)
            # normalize
            sum_weights = weights.sum()
            weights = weights / sum_weights
            # resample
            particle_ind = list(range(N))
            particle_ind = np.random.choice(particle_ind, p=weights, size=N)
            particles = particles[particle_ind, :, :]
            weights = weights[particle_ind]
            particle_ind = np.random.choice(particle_ind, size=N)
            particles_wo_mm = particles_wo_mm[particle_ind, :, :]
          
            particle_poses_all.append(particle_poses)
            particle_poses_all_wo_mm.append(particle_poses_wo_mm)

        np.save(measurement_model_path, particle_poses_all, allow_pickle=False)
        np.save(measurement_model_path+'_wo_mm', particle_poses_all_wo_mm, allow_pickle=False)


if __name__ == '__main__':
    filter = Particle_Filter(10, std_w=2.5e-4**0.5, std_v=4e-3**0.5, gamma=1e-5, bias_w_std=0.9e-9**0.5)
    # 70 to 250
    mapping_indices = list(range(70, 250))
    # 1580 to 1850
    localization_indices = list(range(1580, 1850))
    # localization_indices = list(range(0, 250))
    filter.run(mapping_indices, localization_indices)


