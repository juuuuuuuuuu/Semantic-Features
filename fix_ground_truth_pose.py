import pandas as pd

if __name__ == '__main__':
    path_in = "content/kitti_dataset/dataset/poses/08.txt"
    path_out = "content/kitti_dataset/dataset/poses/08.txt"
    data = pd.read_csv(path_in, header=None, sep=' ')
    print(data.values.shape)
    data.values[:, 7] = 0.
    data.to_csv(path_out, header=None, sep=' ', index=False)


