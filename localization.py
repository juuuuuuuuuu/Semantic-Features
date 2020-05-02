import numpy as np
import time
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
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
    def __init__(self, landmarks, labels, label_colors):
        self.landmarks = landmarks

        self.lm_labels = labels
        self.label_colors = label_colors
        self.pointer = 0
        self.render_single_frame = False
        self.render_pose_connect = False
        self.render_boxes = False

        self.landmark_geometries = render_landmarks(self.landmarks, self.lm_labels, self.label_colors)

        self.ground_grid = render_ground_grid()

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        # vis.register_key_callback(ord("F"), self.get_switch_index_callback(forward=True))
        # vis.register_key_callback(ord("D"), self.get_switch_index_callback(forward=False))
        # vis.register_key_callback(ord("A"), self.get_toggle_show_all_callback())
        # vis.register_key_callback(ord("N"), self.get_toggle_connect_callback())
        vis.register_key_callback(ord("S"), self.get_screen_cap_callback())

        print("Press 'A' to toggle between show all and show only one frame.")
        print("Press 'F' to switch to next frame.")
        print("Press 'D' to switch to previous frame.")
        print("Press 'S' to capture screenshot.")

        for geometry in self.landmark_geometries:
            vis.add_geometry(geometry)

        vis.add_geometry(self.ground_grid)

        vis.run()
        vis.destroy_window()

    def update_render(self, vis):
        view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()

        for geometry in self.landmark_geometries:
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


def render_landmarks(landmarks, labels, label_colors):
    size = 0.5
    size_vec = np.array([size / 2., size / 2., size / 2.])

    boxes = []

    for i in range(len(landmarks)):
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound=landmarks[i, :] - size_vec,
                                                  max_bound=landmarks[i, :] + size_vec)
        box.color = label_colors[labels[i]]
        boxes.append(box)

    return boxes


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
        self.max_dist = 30.
        self.octtree = OctTree(min_bound=np.zeros((3,)), max_bound=np.ones((3,)) * self.max_dist, depth=7)

    def load_landmarks(self, lm_path, label_path):
        # The maximum viewing distance for valid landmarks is lets say 40 m.
        bboxes = np.load(lm_path, allow_pickle=True)
        bboxes = np.vstack(bboxes)
        labels = np.load(label_path, allow_pickle=True)

        print("Number of landmarks: {}".format(bboxes.shape[0]))
        # Use midpoints of merged bounding boxes as landmark.
        self.landmarks = (bboxes[:, :3] + bboxes[:, 3:6]) / 2.
        self.landmark_labels = labels

        # Check how many landmarks below threshold.
        check_dist = 0.5
        num_below = 0
        lms_to_delete = []
        for i in range(self.landmarks.shape[0] - 1):
            for j in range(i + 1, self.landmarks.shape[0]):
                if np.linalg.norm(self.landmarks[i, :] - self.landmarks[j, :]) < check_dist and \
                        self.landmark_labels[i] == self.landmark_labels[j]:
                    num_below += 1
                    if j not in lms_to_delete:
                        lms_to_delete += [j]
        print("Number of landmarks less than {} meters apart with same label is: {}".format(check_dist, num_below))
        print("Number of landmarks deleted: {}".format(len(lms_to_delete)))

        # Remove the landmarks that are too close to another one.
        self.landmarks = np.delete(self.landmarks, lms_to_delete, axis=0)
        self.landmark_labels = np.delete(self.landmark_labels, lms_to_delete)

        print("Number of landmarks after merging: {}".format(self.landmarks.shape[0]))

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
            self.octtree.add(item, item[1][0])

        print("Time to generate oct tree: {}".format(time.perf_counter() - tic))

    def render(self):
        renderer = LandmarkRenderer(self.landmarks, self.landmark_labels, get_colors())
        renderer.run()

    def check_triangle(self):
        tic = time.perf_counter()

        max_dist = 1.5
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

        t_matches = [[] for i in range(self.landmarks.shape[0])]
        lm_matches = [[] for i in range(self.landmarks.shape[0])]
        for i_triangle, item in enumerate(test_triangles.items()):
            for element in self.octtree.query(item[1][0], l1_radius=max_dist):
                matches = [[], [], []]
                if np.linalg.norm(item[1][0] - element[1][0]) <= max_dist:
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





if __name__ == '__main__':
    lol = Map()
    lol.load_landmarks("/home/felix/vision_ws/Semantic-Features/results/mergedbbox.npy",
                       "/home/felix/vision_ws/Semantic-Features/results/classes_list.npy")
    lol.compute_triangles()
    # lol.plot_triangles()
    lol.generate_tree()
    lol.check_triangle()
    lol.render()

