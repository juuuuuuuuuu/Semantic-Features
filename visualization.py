import sys
import os

import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import matplotlib.pyplot as plt


class LandmarkRenderer:
    def __init__(self, poses, landmarks, landmark_pcls, labels, frame_ids, label_colors):
        self.poses = poses
        self.landmarks = landmarks
        self.landmark_pcls = landmark_pcls
        self.lm_labels = labels
        self.lm_frame_ids = frame_ids
        self.label_colors = label_colors
        self.unique_frame_ids = np.unique(frame_ids)
        self.frame_count = len(self.unique_frame_ids)
        self.pointer = 0
        self.render_single_frame = False
        self.render_pose_connect = False
        self.render_boxes = False

        self.landmark_render_objects = render_pcls(self.poses,
                                                   self.landmark_pcls,
                                                   self.lm_labels,
                                                   self.label_colors,
                                                   None)
        self.ground_grid = render_ground_grid()

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        vis.register_key_callback(ord("F"), self.get_switch_index_callback(forward=True))
        vis.register_key_callback(ord("D"), self.get_switch_index_callback(forward=False))
        vis.register_key_callback(ord("A"), self.get_toggle_show_all_callback())
        # vis.register_key_callback(ord("N"), self.get_toggle_connect_callback())
        vis.register_key_callback(ord("S"), self.get_screen_cap_callback())

        print("Press 'A' to toggle between show all and show only one frame.")
        print("Press 'F' to switch to next frame.")
        print("Press 'D' to switch to previous frame.")
        print("Press 'S' to capture screenshot.")

        for geometries in self.landmark_render_objects:
            for geometry in geometries:
                vis.add_geometry(geometry)

        vis.add_geometry(self.ground_grid)

        vis.run()
        vis.destroy_window()

    def update_render(self, vis):
        view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()

        indices = np.where(self.lm_frame_ids == self.unique_frame_ids[self.pointer])

        if self.render_boxes:
            if self.render_single_frame:
                boxes = render_landmarks(
                    self.landmarks[indices, :][0],
                    self.lm_labels[indices, :][0],
                    self.label_colors
                )
                for box in boxes:
                    vis.add_geometry(box)
            else:
                boxes = render_landmarks(self.landmarks, self.lm_labels, self.label_colors)
                for box in boxes:
                    vis.add_geometry(box)

            if self.render_pose_connect:
                if self.render_single_frame:
                    vis.add_geometry(render_pose_connects(
                        self.poses[indices, :][0],
                        self.landmarks[indices, :][0]
                    ))
                else:
                    vis.add_geometry(render_pose_connects(self.poses, self.landmarks))

        if self.render_single_frame:
            for i in range(len(self.landmark_render_objects)):
                if self.lm_frame_ids[i] == self.unique_frame_ids[self.pointer]:
                    for geometry in self.landmark_render_objects[i]:
                        vis.add_geometry(geometry)
        else:
            for geometries in self.landmark_render_objects:
                for geometry in geometries:
                    vis.add_geometry(geometry)

        vis.add_geometry(self.ground_grid)

        # self.get_screen_cap_callback()(vis)

        vis.update_renderer()
        vis.get_view_control().convert_from_pinhole_camera_parameters(view)

    def get_toggle_connect_callback(self):
        def toggle_connect(vis):
            self.render_pose_connect = not self.render_pose_connect
            self.update_render(vis)

        return toggle_connect

    def get_toggle_show_all_callback(self):
        def toggle_show_all(vis):
            self.render_single_frame = not self.render_single_frame
            self.update_render(vis)

        return toggle_show_all

    def get_screen_cap_callback(self):
        def capture_screen(vis):
            image = np.asarray(vis.capture_screen_float_buffer(False))
            path = "results/{}.jpg".format(self.unique_frame_ids[self.pointer])
            cv2.imwrite(path, image * 255., [cv2.IMWRITE_JPEG_QUALITY, 40])
            print("Screenshot saved to " + path)

        return capture_screen

    def get_switch_index_callback(self, forward):
        def switch_index(vis):
            if not self.render_single_frame:
                self.render_single_frame = True
            else:
                if forward:
                    self.pointer = self.pointer + 1
                else:
                    self.pointer = self.pointer - 1
                if self.pointer == self.frame_count:
                    self.pointer = 0
                if self.pointer == -1:
                    self.pointer = self.frame_count - 1

            print("Now showing frame {} ({}/{})".format(
                self.unique_frame_ids[self.pointer], self.pointer + 1, self.frame_count))
            self.update_render(vis)

        return switch_index


def render_landmarks(landmarks, labels, label_colors):
    size = 0.5
    size_vec = np.array([size/2., size/2., size/2.])

    boxes = []

    for i in range(len(landmarks)):
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound=landmarks[i]-size_vec, max_bound=landmarks[i]+size_vec)
        box.color = label_colors[labels[i]]
        boxes.append(box)

    return boxes


def render_pcls(poses, pcls, labels, label_colors, indices):
    geometries = []

    for i in range(len(pcls)):
        if indices and not np.isin(indices, i).any():
            continue

        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.transpose(pcls[i][:3, :]).astype(float))
        )
        pcl.colors = o3d.utility.Vector3dVector([label_colors[labels[i]] for j in range(pcls[i].shape[1])])

        size = 2
        size_vec = np.array([size/2., size/2., size/2.])
        pose_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=poses[i, :]-size_vec, max_bound=poses[i, :]+size_vec)
        pose_box.color = np.array([0.5, 1.0, 0.5])

        geometries.append([pcl, pose_box])

    return geometries


def render_pose_connects(poses, landmarks):
    line_count = poses.shape[0]

    points = np.vstack((poses[:3], landmarks[:3]))
    indices = np.vstack((
                np.hstack((np.arange(line_count).reshape(line_count, 1),
                          (np.arange(line_count) + line_count).reshape(line_count, 1)))
    ))
    colors = [[0.8, 0.8, 0.8] for i in range(line_count*2)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(float).tolist()),
        lines=o3d.utility.Vector2iVector(indices.astype(int).tolist()),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


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


def get_colors():
    colors = np.zeros((500, 3))

    for i in range(500):
        # Generate random numbers. With fixed seeds.
        np.random.seed(i)
        rgb = np.random.randint(255, size=(1, 3)) / 255.0
        colors[i, :] = rgb

    return colors


def load_data(path, n):
    data = pd.read_csv(path, sep=" ", header=None)
    data = data.values
    pcls = []
    labels = []
    frame_ids = []
    poses = []

    for i in range(min(n, data.shape[0])):
        pcls.append(np.load(data[i, 3] + ".npy"))
        labels.append(int(data[i, 2]))
        frame_ids.append(int(data[i, 0]))
        poses.append(data[i, 4:])

    labels = np.array(labels, dtype=int)
    frame_ids = np.array(frame_ids, dtype=int)
    poses = np.array(poses, dtype=float)
    return poses, pcls, labels, frame_ids


def load_lines(path):
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        data_lines = data_lines.values
    except pd.errors.EmptyDataError:
        print("Error, empty data.")
    return data_lines


if __name__ == '__main__':
    path = "results/_results.txt"

    poses, pcls, labels, frame_ids = load_data(path, 1000000)

    #pcl_test = [np.load("pcl_test.npy")]
    #labels_test = np.array([0])
    #frame_ids = np.array([0])

    print("Number of landmarks is: {}".format(labels.shape[0]))
    renderer = LandmarkRenderer(poses, None, pcls, labels, frame_ids, get_colors())
    renderer.run()
