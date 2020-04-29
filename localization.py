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

    def query(self, vector):
        # TODO: Implement proper radius checking.
        if self.depth == 0:
            return self.content
        else:
            x, y, z = self.get_child_id(vector)
            if not self.children[x, y, z]:
                return []
            else:
                return self.children[x, y, z].query(vector)

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


class Map:
    def __init__(self):
        self.landmarks = np.zeros((0, 3))
        self.neighbor_indices = []
        self.triangles = {}
        self.octtree = []
        self.max_dist = 30.

    def load_landmarks(self, path):
        # The maximum viewing distance for valid landmarks is lets say 40 m.
        bboxes = np.load(path, allow_pickle=True)
        bboxes = np.vstack(bboxes)
        print("Number of landmarks: {}".format(bboxes.shape[0]))
        # Use midpoints of merged bounding boxes as landmark.
        self.landmarks = (bboxes[:, :3] + bboxes[:, 3:6]) / 2.

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
                                self.triangles.update({tuple(triangle): vector})

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

    def check_triangle(self):
        max_dist = 4.
        triangle = self.triangles[next(iter(self.triangles))]

        for key, value in self.triangles:
            print("Todo.")


if __name__ == '__main__':
    lol = Map()
    lol.load_landmarks("/home/felix/vision_ws/Semantic-Features/results/mergedbbox.npy")
    lol.compute_triangles()
    lol.plot_triangles()

