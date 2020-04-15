import open3d as o3d


def load_data():
    print("To be implemented.")


import numpy as np
import open3d as o3d
import os
import pandas as pd
import sys


class LandmarkRenderer:
    def __init__(self, poses, landmarks, labels, frame_ids, label_colors):
        self.poses = poses
        self.landmarks = landmarks
        self.lm_labels = labels
        self.lm_frame_ids = frame_ids
        self.label_colors = label_colors
        self.unique_frame_ids = np.unique(frame_ids)
        self.frame_count = len(self.unique_frame_ids)
        self.pointer = 0
        self.render_single_frame = False
        self.render_pose_connect = False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        vis.register_key_callback(ord(" "), self.get_switch_index_callback())
        vis.register_key_callback(ord("A"), self.get_toggle_show_all_callback())
        vis.register_key_callback(ord("N"), self.get_toggle_connect_callback())

        boxes = render_landmarks(self.landmarks, self.lm_labels, self.label_colors)
        for box in boxes:
            vis.add_geometry(box)
        vis.run()
        vis.destroy_window()

    def update_render(self, vis):
        view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()

        indices = np.where(self.lm_frame_ids == self.unique_frame_ids[self.pointer])

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

    def get_switch_index_callback(self):
        def switch_index(vis):
            if not self.render_single_frame:
                self.render_single_frame = True
            else:
                self.pointer = self.pointer + 1
                if self.pointer == self.frame_count:
                    self.pointer = 0

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


def render_pose_connects(poses, landmarks):
    line_count = poses.shape[0]

    mid_points = (lines[:, 0:3] + lines[:, 3:6]) / 2
    ends_1 = mid_points + lines[:, 7:10] * 0.05
    ends_2 = mid_points + lines[:, 10:13] * 0.05

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


def get_colors():
    colors = np.zeros((500, 3))

    for i in range(500):
        # Generate random numbers. With fixed seeds.
        np.random.seed(i)
        rgb = np.random.randint(255, size=(1, 3)) / 255.0
        colors[i, :] = rgb

    return colors


def load_data(path):
    print("Loading data from path {}".format(path))
    print("TODO: Implement.")


def load_lines(path):
    try:
        data_lines = pd.read_csv(path, sep=" ", header=None)
        data_lines = data_lines.values
    except pd.errors.EmptyDataError:
        print("Error, empty data.")
    return data_lines


if __name__ == '__main__':
    path = "results.json"

    #data = load_data()

    lines = load_lines("/home/felix/line_ws/data/line_tools/interiornet_lines_split/all_lines_with_line_endpoints.txt")
    landmarks = lines[:, [1, 2, 3]]
    labels = lines[:, 7]

    print("Number of landmarks is: {}".format(lines.shape[0]))
    renderer = LandmarkRenderer(landmarks, landmarks, labels, labels, get_colors())
    renderer.run()
