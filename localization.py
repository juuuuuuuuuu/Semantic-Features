import numpy as np
import time
import matplotlib.pyplot as plt


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
        check_dist = 0.4
        num_below = 0
        lms_to_delete = []
        for i in range(self.landmarks.shape[0] - 1):
            for j in range(i + 1, self.landmarks.shape[0]):
                if np.linalg.norm(self.landmarks[i, :] - self.landmarks[j, :]) < check_dist:# and \
                        # self.landmark_labels[i] == self.landmark_labels[j]:
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
                                vector = np.sort(vector)
                                self.triangles.update({triangle_key: vector})

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
            self.octtree.add(item, item[1])

        print("Time to generate oct tree: {}".format(time.perf_counter() - tic))

    def check_triangle(self):
        max_dist = 0.4
        num_landmarks = 8

        anchor = 0
        test_nbh = [anchor]
        for i in range(1, self.landmarks.shape[0]):
            if np.linalg.norm(self.landmarks[anchor, :] - self.landmarks[i, :]) < self.max_dist / 2.:
                test_nbh += [i]

            if len(test_nbh) >= num_landmarks:
                break

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
                                vector = np.sort(vector)
                                test_triangles.update({triangle_key: vector})

        print("Landmark count is: {}".format(len(test_nbh)))
        print("Test triangle count is: {}".format(len(test_triangles)))

        matches = [[] for i in range(self.landmarks.shape[0])]
        for i_triangle, item in enumerate(test_triangles.items()):
            for element in self.octtree.query(item[1], l1_radius=max_dist/2.):
                if np.linalg.norm(item[1] - element[1]) <= max_dist:
                    for index in element[0]:
                        if i_triangle not in matches[index]:
                            matches[index] += [i_triangle]

        match_counts = np.array([len(match) for match in matches])
        print("Match counts:")
        print(np.sort(match_counts)[-15:])
        print("Predicted landmark indices:")
        print(np.argsort(match_counts)[-15:])
        print("Actual landmark indices:")
        print(test_nbh)

        # triangle_key = next(iter(self.triangles))
        # triangle_vec = self.triangles[triangle_key]
        #
        # print("Triangle selected is connected to: {}".format(triangle_key))
        # print("It has side lengths: {}".format(triangle_vec))
        # print(" ")
        # print("In the same cell of this triangle there are the following triangles: ")
        # print(" ")
        #
        # tic = time.perf_counter()
        # for element in self.octtree.query(triangle_vec, l1_radius=max_dist/2.):
        #     print("Landmarks: {}".format(element[0]))
        #     print("Side lengths: {}".format(element[1]))
        #     print("Distance to original is: ".format(np.linalg.norm(element[1] - triangle_vec)))
        #
        # print("This query took {} seconds.".format(time.perf_counter() - tic))


if __name__ == '__main__':
    lol = Map()
    lol.load_landmarks("/home/felix/vision_ws/Semantic-Features/results/mergedbbox.npy",
                       "/home/felix/vision_ws/Semantic-Features/results/classes_list.npy")
    lol.compute_triangles()
    # lol.plot_triangles()
    lol.generate_tree()
    lol.check_triangle()

