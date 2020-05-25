import pandas as pd
import numpy as np
import scipy.spatial.transform.rotation as r

if __name__ == '__main__':
    path_in = "content/kitti_dataset/dataset/poses/08.txt"
    path_out = "content/kitti_dataset/dataset/poses/08.txt"
    data = pd.read_csv(path_in, header=None, sep=' ')
    print(data.values.shape)
    data.values[:, 7] = 0.

    for i in range(data.values.shape[0]):
        rot_mat = np.vstack([data.values[i, :3], data.values[i, 4:7], data.values[i, 8:11]])
        rot_mat = r.Rotation.from_matrix(rot_mat)
        rot_vec = rot_mat.as_rotvec()
        rot_vec[0] = 0.
        rot_vec[2] = 0.
        new_rot_mat = r.Rotation.from_rotvec(rot_vec).as_matrix()
        data.values[i, :3] = new_rot_mat[0, :3]
        data.values[i, 4:7] = new_rot_mat[1, :3]
        data.values[i, 8:11] = new_rot_mat[2, :3]

    data.to_csv(path_out, header=None, sep=' ', index=False)


