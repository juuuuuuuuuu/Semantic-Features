import numpy as np
import time
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
import random
import pandas as pd
import scipy.linalg
from collections import Counter


def get_colors():
    colors = np.zeros((500, 3))

    for i in range(500):
        # Generate random numbers. With fixed seeds.
        np.random.seed(i)
        rgb = np.random.randint(255, size=(1, 3)) / 255.0
        colors[i, :] = rgb

    return colors


class LandmarkRenderer:
    def __init__(self, landmarks, labels, all_landmarks, all_labels, frame_lms, frame_labels, pose_est, pose_gts,
                 query_inliers, label_colors):
        self.landmarks = landmarks
        self.lm_labels = labels

        self.all_landmarks = all_landmarks
        self.all_lm_labels = all_labels

        self.frame_lms = frame_lms
        self.frame_labels = frame_labels
        self.label_colors = label_colors
        self.pointer = 0
        self.frame_count = len(frame_lms)
        self.render_single_frame = False
        self.render_pose_connect = False
        self.render_boxes = False
        self.render_all = False
        self.render_matched = False

        self.poses_gts_geometries = [render_pose(pose, False) for pose in pose_gts]
        self.poses_est_geometries = [render_pose(pose, True) for pose in pose_est]
        self.frame_matches = [render_matches(frame_lms[i], landmarks, query_inliers[i], pose_est[i]) for i
                              in range(len(frame_lms))]
        self.frame_lms_geometries = [render_landmarks(frame_lms[i], frame_labels[i], self.label_colors, size=0.1) for i
                                     in range(self.frame_count)]

        self.landmark_geometries = render_landmarks(self.landmarks, self.lm_labels, self.label_colors)
        self.all_landmark_geometries = render_landmarks(self.all_landmarks, self.all_lm_labels, self.label_colors)

        self.ground_grid = render_ground_grid()

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        vis.register_key_callback(ord("F"), self.get_switch_index_callback(forward=True))
        vis.register_key_callback(ord("D"), self.get_switch_index_callback(forward=False))
        vis.register_key_callback(ord("A"), self.get_toggle_show_all_callback())
        vis.register_key_callback(ord("M"), self.get_toggle_matched_callback())
        vis.register_key_callback(ord("S"), self.get_screen_cap_callback())

        print("Press 'A' to toggle between show merged or show all landmarks.")
        print("Press 'F' to switch to next frame.")
        print("Press 'D' to switch to previous frame.")
        print("Press 'S' to capture screenshot.")

        for geometry in self.landmark_geometries:
            vis.add_geometry(geometry)

        for geometry in self.frame_lms_geometries[self.pointer]:
            vis.add_geometry(geometry)

        for geometry in self.poses_gts_geometries:
            vis.add_geometry(geometry)

        for geometry in self.poses_est_geometries:
            if geometry:
                vis.add_geometry(geometry)

        vis.add_geometry(self.ground_grid)

        vis.run()
        vis.destroy_window()

    def update_render(self, vis):
        view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()

        if self.render_all:
            for geometry in self.all_landmark_geometries:
                vis.add_geometry(geometry)
        else:
            for geometry in self.landmark_geometries:
                vis.add_geometry(geometry)

        if self.render_matched:
            for geometry in self.frame_lms_geometries[self.pointer]:
                vis.add_geometry(geometry)
            vis.add_geometry(self.poses_gts_geometries[self.pointer])
            if self.frame_matches[self.pointer]:
                vis.add_geometry(self.frame_matches[self.pointer])
                vis.add_geometry(self.poses_est_geometries[self.pointer])
        else:
            for geometry in self.poses_gts_geometries:
                vis.add_geometry(geometry)
            for geometry in self.poses_est_geometries:
                if geometry:
                    vis.add_geometry(geometry)

        vis.add_geometry(self.ground_grid)

        # self.get_screen_cap_callback()(vis)

        vis.update_renderer()
        vis.get_view_control().convert_from_pinhole_camera_parameters(view)

    def get_toggle_matched_callback(self):
        def toggle_matched(vis):
            self.render_matched = not self.render_matched
            self.update_render(vis)

        return toggle_matched

    def get_toggle_show_all_callback(self):
        def toggle_show_all(vis):
            self.render_all = not self.render_all
            self.update_render(vis)

        return toggle_show_all

    def get_screen_cap_callback(self):
        def capture_screen(vis):
            exit(0)

            image = np.asarray(vis.capture_screen_float_buffer(False))
            path = "results/{}.jpg".format(self.unique_frame_ids[self.pointer])
            cv2.imwrite(path, image * 255., [cv2.IMWRITE_JPEG_QUALITY, 40])
            print("Screenshot saved to " + path)

        return capture_screen

    def get_switch_index_callback(self, forward):
        def switch_index(vis):
            if forward:
                self.pointer = self.pointer + 1
            else:
                self.pointer = self.pointer - 1
            if self.pointer == self.frame_count:
                self.pointer = 0
            if self.pointer == -1:
                self.pointer = self.frame_count - 1

            print("Now showing frame {}/{}".format(self.pointer + 1, self.frame_count))
            self.update_render(vis)

        return switch_index


def render_ground_grid():
    scale = 10
    size = 30
    count = size * 2
    x_min = -10
    x_max = 2 * size - 10
    x = np.arange(x_min, x_max).reshape(count, 1) * scale
    y = np.arange(-size, size).reshape(count, 1) * scale
    z = -2 * np.ones((count, 1))
    x_1 = np.hstack([x, y * 0. + size * scale, z])
    x_2 = np.hstack([x, y * 0. - size * scale, z])
    y_1 = np.hstack([x * 0. + x_max * scale, y, z])
    y_2 = np.hstack([x * 0. + x_min * scale, y, z])
    points = np.vstack([x_1, x_2, y_1, y_2])
    indices = np.vstack([
        np.hstack([np.arange(count).reshape(count, 1), (np.arange(count) + count).reshape(count, 1)]),
        np.hstack([(np.arange(count) + 2 * count).reshape(count, 1), (np.arange(count) + 3 * count).reshape(count, 1)])
    ])
    colors = [[0.3, 0.3, 0.3] for i in range(count * 2)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def render_landmarks(landmarks, labels, label_colors, size=0.5):
    size_vec = np.array([size / 2., size / 2., size / 2.])

    boxes = []
    for i in range(len(landmarks)):
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound=landmarks[i, :] - size_vec,
                                                  max_bound=landmarks[i, :] + size_vec)
        box.color = label_colors[labels[i]]
        boxes.append(box)

    return boxes


def render_matches(frame_landmarks, map_landmarks, matches, pose):
    if matches is None:
        return None

    line_count = matches.shape[0]

    points = np.vstack((frame_landmarks[matches[:, 0], :],
                        map_landmarks[matches[:, 1], :],
                        pose))
    indices = np.vstack((
        np.hstack((np.arange(line_count).reshape(line_count, 1),
                   np.arange(line_count, line_count*2).reshape(line_count, 1))),
        np.hstack((np.ones((line_count, 1), dtype=int) * 2 * line_count,
                   np.arange(line_count).reshape(line_count, 1)))))

    colors = [[1.0, 0.5, 0.5] for i in range(line_count * 2)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def render_pose(pose, red):
    if pose is None:
        return None

    size = 1.
    size_vec = np.array([size/2., size/2., size/2.])
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=pose - size_vec,
                                              max_bound=pose + size_vec)
    if red:
        box.color = np.array([1.0, 0.1, 0.1])
    else:
        box.color = np.array([0.3, 1.0, 0.3])

    return box


class OctTree:
    def __init__(self, min_bound, max_bound, depth):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.size = max_bound - min_bound
        self.middle = (min_bound + max_bound) / 2.
        self.content = []
        self.depth = depth
        self.children = np.zeros((2, 2, 2), dtype=OctTree)

    def add(self, element, vector):
        if self.depth == 0:
            self.content.append(element)
        else:
            x, y, z = self.get_child_id(vector)
            if not self.children[x, y, z]:
                self.children[x, y, z] = OctTree(self.min_bound + np.array([self.size[0] / 2. * x,
                                                                            self.size[1] / 2. * y,
                                                                            self.size[2] / 2. * z]),
                                                 self.min_bound + np.array([self.size[0] / 2. * (x + 1),
                                                                            self.size[1] / 2. * (y + 1),
                                                                            self.size[2] / 2. * (z + 1)]),
                                                 self.depth - 1)
            self.children[x, y, z].add(element, vector)

    def query(self, vector, l1_radius=0.):
        if not self.check_contains(vector, l1_radius):
            return []
        elif self.depth == 0:
            return self.content
        else:
            elements = []
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        if self.children[x, y, z]:
                            elements = elements + self.children[x, y, z].query(vector, l1_radius)
            return elements

    def get_child_id(self, vector):
        x = 0
        y = 0
        z = 0
        if vector[0] > self.middle[0]:
            x = 1
        if vector[1] > self.middle[1]:
            y = 1
        if vector[2] > self.middle[2]:
            z = 1

        return x, y, z

    def check_contains(self, vector, l1_radius=0.):
        radius_vec = np.ones(3,) * l1_radius
        return (self.min_bound - radius_vec < vector).all() and (vector < self.max_bound + radius_vec).all()


class Map:
    def __init__(self):
        self.landmarks = np.zeros((0, 3))
        self.landmark_labels = np.zeros((0, 1))
        self.neighbor_indices = []
        self.triangles = {}
        # The maximum viewing distance for valid landmarks is lets say 50 m.
        self.max_dist = 50.
        self.oct_trees = {}
        self.all_landmarks = []
        self.all_landmark_labels = []

    def load_landmarks(self, frame_lms, frame_labels):
        map_landmarks = np.vstack(frame_lms)
        map_landmark_labels = np.concatenate(frame_labels, axis=0)

        # Safe old landmarks for later.
        self.all_landmarks = np.copy(map_landmarks)
        self.all_landmark_labels = np.copy(map_landmark_labels)

        # Set label 1 (green) and label 15 (yellow) as same label (they often appear together)
        map_landmark_labels[np.where(map_landmark_labels == 1)] = 15

        n_landmarks_before = map_landmarks.shape[0]
        print("Number of map landmarks: {}".format(map_landmarks.shape[0]))
        # Use midpoints of merged bounding boxes as landmark.
        # self.landmarks = (bboxes[:, :3] + bboxes[:, 3:6]) / 2.
        # self.landmark_labels = labels

        # Mean shift clustering of landmarks for merging.
        mean_shift_radius = 1.5
        max_iters = 20

        new_landmarks = []
        new_landmark_labels = []

        while map_landmarks.shape[0] > 0:
            mean = map_landmarks[0, :]

            n_iters = 0
            while True:
                n_iters += 1
                label = -1
                lms_in_radius = []
                new_mean = np.zeros((3,))
                for i in range(0, map_landmarks.shape[0]):
                    if np.linalg.norm(map_landmarks[i, :] - mean) < mean_shift_radius:
                        if label == -1:
                            label = map_landmark_labels[i]
                        elif label != map_landmark_labels[i]:
                            continue
                        lms_in_radius.append(i)
                        new_mean += map_landmarks[i, :]

                new_mean = new_mean / len(lms_in_radius)

                if np.equal(new_mean, mean).all() or n_iters >= max_iters:
                    # print("Deleted {}".format(lms_in_radius))
                    # print("Landmarks left: {}".format(map_landmarks.shape[0]))
                    map_landmarks = np.delete(map_landmarks, lms_in_radius, axis=0)
                    map_landmark_labels = np.delete(map_landmark_labels, lms_in_radius, axis=0)
                    new_landmarks.append(new_mean)
                    new_landmark_labels.append(label)
                    break
                else:
                    mean = new_mean

        print("Number of map landmarks deleted with merging: {}".format(n_landmarks_before - len(new_landmarks)))
        self.landmarks = np.vstack(new_landmarks)
        self.landmark_labels = np.array(new_landmark_labels)

        # Find the neighbors within range of each landmark:
        for i in range(self.landmarks.shape[0]):
            neighbour_indices = []
            # This can be optimized by removing the ones already handled.
            for j in range(self.landmarks.shape[0]):
                if not i == j:
                    if np.linalg.norm(self.landmarks[i, :] - self.landmarks[j, :]) < self.max_dist:
                        neighbour_indices.append(j)

            self.neighbor_indices.append(neighbour_indices)

        max_neighbor_count = 0
        average_nb_count = 0.
        for i in range(self.landmarks.shape[0]):
            nb_count = len(self.neighbor_indices[i])
            if nb_count > max_neighbor_count:
                max_neighbor_count = nb_count

            average_nb_count += nb_count

        average_nb_count = average_nb_count / self.landmarks.shape[0]

        print("Maximum neighbor count: {}".format(max_neighbor_count))
        print("Average neighbor count: {}".format(average_nb_count))

    def compute_triangles(self):
        print("Computing triangles.")
        tic = time.perf_counter()

        # The minimum side length of a triangle, to prevent over matching (low side lengths are very common).
        min_side_length = 1.

        # This is done to sort triangle indices according to their length. (Right hand rule)
        # Found no better way than to hack the shit out of this.
        order_mapping = {
            (0, 1, 2): (0, 1, 2),
            (0, 2, 1): (1, 0, 2),
            (1, 0, 2): (2, 1, 0),
            (1, 2, 0): (1, 2, 0),
            (2, 0, 1): (2, 0, 1),
            (2, 1, 0): (0, 2, 1)
        }

        for i in range(self.landmarks.shape[0]):
            for nb_1 in self.neighbor_indices[i]:
                length_1 = np.linalg.norm(self.landmarks[i, :] - self.landmarks[nb_1, :])
                for nb_2 in self.neighbor_indices[i]:
                    # Dirty...
                    if nb_2 > nb_1:
                        length_2 = np.linalg.norm(self.landmarks[nb_2, :] - self.landmarks[nb_1, :])
                        if length_2 < self.max_dist:
                            triangle = [i, nb_1, nb_2]
                            triangle.sort()
                            triangle_key = tuple(triangle)
                            if triangle_key not in self.triangles:
                                vector = np.array([
                                    length_1,
                                    length_2,
                                    np.linalg.norm(self.landmarks[i, :] - self.landmarks[nb_2, :])
                                ])
                                order = np.argsort(vector)
                                vector = vector[order]
                                if vector[-1] > min_side_length:
                                    order_map = order_mapping[tuple(order)]
                                    lms = [triangle[order_map[i]] for i in range(3)]
                                    self.triangles.update({triangle_key: [vector, lms]})

        print("Triangle count is: {}".format(len(self.triangles)))
        toc = time.perf_counter()
        print("Took {} seconds to compute.".format(toc-tic))
        print("Average time per triangle: {}".format((toc-tic)/len(self.triangles)))

    def plot_triangles(self):
        points = np.array([value for value in self.triangles.values()])
        print(points.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:10000, 0], points[:10000, 1], points[:10000, 2])
        plt.show()

    def generate_tree(self):
        tic = time.perf_counter()
        for item in self.triangles.items():
            # item: {triangle_key, [vector, landmarks]}
            labels_key = tuple([self.landmark_labels[item[1][1][i]] for i in range(3)])
            if labels_key not in self.oct_trees:
                self.oct_trees.update({labels_key: OctTree(min_bound=np.zeros((3,)),
                                                           max_bound=np.ones((3,)) * self.max_dist,
                                                           depth=7)})
            self.oct_trees[labels_key].add(item, item[1][0])

        print("Time to generate oct tree: {}".format(time.perf_counter() - tic))

    def render(self, frame_landmarks, frame_labels, pose_estimations, pose_gts, inliers):
        renderer = LandmarkRenderer(self.landmarks, self.landmark_labels,
                                    self.all_landmarks, self.all_landmark_labels,
                                    frame_landmarks, frame_labels,
                                    pose_estimations, pose_gts,
                                    inliers, get_colors())
        renderer.run()

    def check_triangle(self):
        tic = time.perf_counter()

        max_dist = 2.0
        num_landmarks = 8

        anchor = 0
        test_nbh = [anchor]
        for i in range(1, self.landmarks.shape[0]):
            if np.linalg.norm(self.landmarks[anchor, :] - self.landmarks[i, :]) < self.max_dist / 2.:
                test_nbh += [i]

            if len(test_nbh) >= num_landmarks:
                break

        # This is done to sort triangle indices according to their length. (Right hand rule)
        # Found no better way than to hack the shit out of this.
        order_mapping = {
            (0, 1, 2): (0, 1, 2),
            (0, 2, 1): (1, 0, 2),
            (1, 0, 2): (2, 1, 0),
            (1, 2, 0): (1, 2, 0),
            (2, 0, 1): (2, 0, 1),
            (2, 1, 0): (0, 2, 1)
        }
        test_triangles = {}
        for i in test_nbh:
            for nb_1 in test_nbh:
                if nb_1 == i:
                    continue
                length_1 = np.linalg.norm(self.landmarks[i, :] - self.landmarks[nb_1, :])
                if length_1 > self.max_dist:
                    continue
                for nb_2 in test_nbh:
                    if nb_2 == i:
                        continue
                    # Dirty...
                    if nb_2 > nb_1:
                        length_2 = np.linalg.norm(self.landmarks[nb_2, :] - self.landmarks[nb_1, :])
                        if length_2 < self.max_dist:
                            triangle = [i, nb_1, nb_2]
                            triangle.sort()
                            triangle_key = tuple(triangle)
                            if triangle_key not in test_triangles:
                                vector = np.array([
                                    length_1,
                                    length_2,
                                    np.linalg.norm(self.landmarks[i, :] - self.landmarks[nb_2, :])
                                ])
                                order = np.argsort(vector)
                                vector = vector[order]
                                order_map = order_mapping[tuple(order)]
                                lms = [triangle[order_map[i]] for i in range(3)]
                                test_triangles.update({triangle_key: [vector, lms]})

        print("Landmark count is: {}".format(len(test_nbh)))
        print("Test triangle count is: {}".format(len(test_triangles)))

        triangles_matched = 0
        max_tris_matched = 0

        t_matches = [[] for i in range(self.landmarks.shape[0])]
        lm_matches = [[] for i in range(self.landmarks.shape[0])]
        for i_triangle, item in enumerate(test_triangles.items()):
            labels_key = tuple([self.landmark_labels[item[1][1][i]] for i in range(3)])
            triangles_matched_here = 0
            for element in self.oct_trees[labels_key].query(item[1][0], l1_radius=max_dist):
                matches = [[], [], []]
                if np.linalg.norm(item[1][0] - element[1][0]) <= max_dist:
                    triangles_matched_here += 1

                    for i in range(3):
                        lm_index_world = element[1][1][i]
                        if lm_index_world not in matches[i]:
                            matches[i] += [lm_index_world]

                for i in range(3):
                    for lm in matches[i]:
                        lm_index_frame = item[1][1][i]
                        lm_matches[lm] += [lm_index_frame]
                        # if i_triangle not in t_matches[lm_index]:
                        #     t_matches[lm_index] += [i_triangle]
                        #     lm_matches[lm_index] += [item[1][1][i]]

            triangles_matched += triangles_matched_here
            max_tris_matched = max(max_tris_matched, triangles_matched_here)

        print("Average amount of map triangles matched per frame triangle: {}".format(
            triangles_matched / len(test_triangles)))
        print("Maximum amount of map triangles matched per one frame triangle: {}".format(
            max_tris_matched))

        match_counts = np.zeros((self.landmarks.shape[0],))
        match_indices = np.zeros((self.landmarks.shape[0],))
        match_dict = {}
        for i, match in enumerate(lm_matches):
            most_common = Counter(match).most_common(1)
            if most_common:
                match_counts[i] = most_common[0][1]
                match_indices[i] = most_common[0][0]

            match_dict.update({i: [match_indices[i], match_counts[i]]})

        argsort = np.argsort(match_counts)
        out = np.stack([argsort, match_indices[argsort], match_counts[argsort]])
        print("Match counts:")
        print(np.transpose(out[:, -30:]))

        print("This query took {} seconds.".format(time.perf_counter() - tic))

    def query_frame(self, landmarks, labels):
        tic = time.perf_counter()

        labels[np.where(labels == 1)] = 15

        # This is done to sort triangle indices according to their length. (Right hand rule)
        # Found no better way than to hack the shit out of this.
        order_mapping = {
            (0, 1, 2): (0, 1, 2),
            (0, 2, 1): (1, 0, 2),
            (1, 0, 2): (2, 1, 0),
            (1, 2, 0): (1, 2, 0),
            (2, 0, 1): (2, 0, 1),
            (2, 1, 0): (0, 2, 1)
        }

        max_dist = 1.0

        maximum_length = 0
        frame_triangles = {}
        for i in range(landmarks.shape[0]):
            for nb_1 in range(i + 1, landmarks.shape[0]):
                length_1 = np.linalg.norm(landmarks[i, :] - landmarks[nb_1, :])
                for nb_2 in range(landmarks.shape[0]):
                    if nb_2 == i or nb_2 == nb_1:
                        continue

                    length_2 = np.linalg.norm(landmarks[nb_2, :] - landmarks[nb_1, :])

                    triangle = [i, nb_1, nb_2]
                    triangle.sort()
                    triangle_key = tuple(triangle)
                    if triangle_key not in frame_triangles:
                        vector = np.array([
                            length_1,
                            length_2,
                            np.linalg.norm(landmarks[i, :] - landmarks[nb_2, :])
                        ])
                        order = np.argsort(vector)
                        vector = vector[order]
                        order_map = order_mapping[tuple(order)]
                        lms = [triangle[order_map[i]] for i in range(3)]
                        frame_triangles.update({triangle_key: [vector, lms]})
                        maximum_length = max(maximum_length, np.max(vector))

        print("Test landmark count is: {}".format(landmarks.shape[0]))
        print("Test triangle count is: {}".format(len(frame_triangles)))
        print("Maximum triangle side length is: {}".format(maximum_length))

        triangles_matched = 0
        max_tris_matched = 0

        t_matches = [[] for i in range(self.landmarks.shape[0])]
        lm_matches = [[] for i in range(self.landmarks.shape[0])]
        for i_triangle, item in enumerate(frame_triangles.items()):
            labels_key = tuple([labels[item[1][1][i]] for i in range(3)])
            triangles_matched_here = 0
            if labels_key in self.oct_trees:
                for element in self.oct_trees[labels_key].query(item[1][0], l1_radius=max_dist):
                    matches = [[], [], []]
                    if np.linalg.norm(item[1][0] - element[1][0]) <= max_dist:
                        triangles_matched_here += 1

                        for i in range(3):
                            lm_index_world = element[1][1][i]
                            if lm_index_world not in matches[i]:
                                matches[i] += [lm_index_world]

                    for i in range(3):
                        for lm in matches[i]:
                            lm_index_frame = item[1][1][i]
                            lm_matches[lm] += [lm_index_frame]
                            # if i_triangle not in t_matches[lm_index]:
                            #     t_matches[lm_index] += [i_triangle]
                            #     lm_matches[lm_index] += [item[1][1][i]]

            triangles_matched += triangles_matched_here
            max_tris_matched = max(max_tris_matched, triangles_matched_here)

        print("Average amount of map triangles matched per frame triangle: {}".format(
            triangles_matched / len(frame_triangles)))
        print("Maximum amount of map triangles matched per one frame triangle: {}".format(
            max_tris_matched))

        match_counts = np.zeros((self.landmarks.shape[0],))
        match_indices = np.zeros((self.landmarks.shape[0],))
        match_dict = {}
        for i, match in enumerate(lm_matches):
            most_common = Counter(match).most_common(1)
            if most_common:
                match_counts[i] = most_common[0][1]
                match_indices[i] = most_common[0][0]

            match_dict.update({i: [match_indices[i], match_counts[i]]})

        argsorted = np.argsort(match_counts)
        out = np.stack([match_indices[argsorted], argsorted, match_counts[argsorted]]).astype(int)
        print("Match counts:")
        print(np.transpose(np.flip(out, axis=1)[:, :30]))

        # self.matched_lms = argsorted[-30:]
        matches = np.transpose(np.flip(out, axis=1))

        transform, inliers = self.odometry_ransac(matches, landmarks)

        print("This query took {} seconds.".format(time.perf_counter() - tic))

        return transform, inliers

    def odometry_ransac(self, matches, frame_landmarks):
        # Number of best matches to be considered in the query.
        match_count = frame_landmarks.shape[0] * 2

        # Minimum number of inliers for proposal.
        min_inlier_support = max(6, int(frame_landmarks.shape[0] / 2))

        # Maximum amount of deviation from actual positions.
        max_error = 2.0

        # Max number of iterations to find inliers with a probability of p = 0.99
        # under the assumption that the probability to find a inlier starting triangle is e = 0.5
        # e = p_triangle_exists * p_correct_heap = 0.5 * 0.5
        p_ = 0.99
        e_ = 0.5 * 0.25 * 0.1
        max_iterations = int(np.log(1 - p_) / np.log(1 - (e_)))

        print("Starting RANSAC, max iterations is {}.".format(max_iterations))
        num_iterations = 0

        matched_frame_lms = frame_landmarks[matches[:match_count, 0], :]
        matched_map_lms = self.landmarks[matches[:match_count, 1], :]

        # The attempted first matches.
        attempts = []
        while True:
            num_iterations += 1

            # Find a starting attempt where all frame indices are different (otherwise we don't get a triangle):
            # Hopefully this will never be an endless loop...
            attempt = random.sample(range(match_count), 3)
            while np.unique(matches[attempt, 0]).shape[0] < 3:
                attempt = random.sample(range(match_count), 3)

            # print(matched_frame_lms[attempt, :])
            proposal_T = triangle_transform(matched_frame_lms[attempt, :].T, matched_map_lms[attempt, :].T)

            transformed_frame_lms = np.dot(proposal_T, np.vstack((matched_frame_lms.T, np.ones((1, match_count))))).T[:, :3]
            difference = np.linalg.norm(transformed_frame_lms - matched_map_lms, axis=-1)

            # print(transformed_frame_lms)
            # print(matched_map_lms)

            # print(difference)
            inliers = np.where(difference < max_error)
            num_inliers = inliers[0].shape[0]

            if num_inliers >= min_inlier_support:
                final_T = triangle_transform(matched_frame_lms[inliers[0], :].T, matched_map_lms[inliers[0], :].T)
                print("Finish RANSAC after {} iterations with {} inliers.".format(num_iterations, num_inliers))
                return final_T, matches[inliers[0], :]

            if num_iterations >= max_iterations:
                print("Not enough inliers found. Exiting.")
                break

        return None, None


def triangle_transform(a, b):
    # a and b are column vectors. a corresponds to the moving frame, b to the inertial frame.
    # WARNING: a and b have to be a horizontal array of column vectors.
    a_mean = np.mean(a, axis=-1, keepdims=True)
    b_mean = np.mean(b, axis=-1, keepdims=True)
    # p_b_minus_a = (b_mean - a_mean).reshape(3, 1)

    x_1, _, _, _ = np.linalg.lstsq((a - a_mean).T, (b - b_mean)[0, :].T, rcond=None)
    x_2, _, _, _ = np.linalg.lstsq((a - a_mean).T, (b - b_mean)[1, :].T, rcond=None)
    x_3, _, _, _ = np.linalg.lstsq((a - a_mean).T, (b - b_mean)[2, :].T, rcond=None)

    R = np.hstack([x_1.reshape(3, 1), x_2.reshape(3, 1), x_3.reshape(3, 1)]).T

    if np.isnan(R).any() or np.isinf(R).any():
        print("Warning, R has inf or nan values.")
        return np.eye(4)

    # Make R orthogonal (rotation matrix, not scale)
    try:
        R = scipy.linalg.sqrtm(R.T.dot(R))
        if np.isnan(R).any() or np.isinf(R).any():
            print("Warning, R has inf or nan values.")
            return np.eye(4)

        R = R.dot(scipy.linalg.inv(R))
    except scipy.linalg.LinAlgError:
        print("Warning, singular matrix.")
        return np.eye(4)

    p_w_f = b_mean + np.dot(R, -a_mean)

    return np.vstack((np.hstack((R, p_w_f)), np.array([0., 0., 0., 1.])))


def load_frame(path, frame_indices):
    results = pd.read_csv(path, header=None, sep=' ')
    data = np.array(results.values)
    frame_landmarks = []
    frame_labels = []
    poses = []
    for i in frame_indices:
        frame_data = data[np.where(data[:, 0] == i), :][0, :, :]
        labels = frame_data[:, 2]
        frame_paths = frame_data[:, 4]
        pose = frame_data[0, 5:8].astype(float)
        landmarks = np.zeros((len(frame_paths), 3))
        for j, f_path in enumerate(frame_paths):
            bbox = np.load(f_path + '.npy')[:, 1]
            landmarks[j, :] = (bbox[:3] + bbox[3:6]) / 2.
        frame_landmarks.append(landmarks)
        frame_labels.append(labels)
        poses.append(pose)

    return poses, frame_landmarks, frame_labels


if __name__ == '__main__':
    map = Map()

    FRAME_COUNT = 271

    frame_list = list(range(FRAME_COUNT))
    poses, f_lms, f_labels = load_frame("/home/felix/vision_ws/Semantic-Features/results/_results.txt", frame_list)

    # Use every third frame for localization, all other frames for mapping
    frames_for_localization = list(range(0, FRAME_COUNT, 3))
    frames_for_mapping = [i for i in frame_list if i not in frames_for_localization]

    map.load_landmarks([f_lms[i] for i in frames_for_mapping], [f_labels[i] for i in frames_for_mapping])
    map.compute_triangles()
    # map.plot_triangles()
    map.generate_tree()

    # alph = 0.5
    # rot = np.array([[np.cos(alph), -np.sin(alph), 0], [np.sin(alph), np.cos(alph), 0], [0., 0., 1.]])
    rot = np.eye(3)
    # off = np.array([10.5, 5.2, 0.4])
    pose_estimations = []
    pose_gts = []
    inliers = []
    frame_lms_loc = []
    frame_labels_loc = []

    num_matched = 0
    square_error = 0

    for i in frames_for_localization:
        if f_lms[i].shape[0] >= 6:
            # Transform frames back into "camera frame" for query.
            transformation, query_inliers = map.query_frame(np.dot(rot, f_lms[i].T).T - poses[i].reshape(1, 3), f_labels[i])
            if transformation is None:
                pose_estimations.append(None)
            else:
                pose_estimations.append(transformation[:3, 3])
                num_matched += 1
                square_error += np.linalg.norm(transformation[:3, 3] - poses[i]) ** 2
            inliers.append(query_inliers)
        else:
            print("This frame has less than 6 instances - skip.")
            pose_estimations.append(None)
            inliers.append(None)

        frame_lms_loc.append(f_lms[i])
        frame_labels_loc.append(f_labels[i])
        pose_gts.append(poses[i])

    print("")
    print("===================================================================")
    print("")
    print("Number of poses correctly matched is {} out of {}.".format(num_matched, len(pose_gts)))
    print("RMSE: {}".format(np.sqrt(square_error / num_matched)))
    print("")
    print("===================================================================")
    print("")

    map.render(frame_lms_loc, frame_labels_loc, pose_estimations, pose_gts, inliers)
    # print(rot)
    # map.check_triangle()

