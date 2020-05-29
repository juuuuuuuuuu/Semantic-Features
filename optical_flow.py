import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


def calculate_flow(previous_im, next_im):
    flow = cv2.calcOpticalFlowFarneback(next_im, previous_im, None, 0.3, 3, 15, 3, 3, 1.2, 0)

    return flow


def update_mask(prob_mask, instance_im, flow, weight):
    map_1 = np.squeeze(flow[:, :, 0].astype(np.float32))
    map_2 = np.squeeze(flow[:, :, 1].astype(np.float32))
    map_1 += np.expand_dims(np.arange(flow.shape[1]), 0)
    map_2 += np.expand_dims(np.arange(flow.shape[0]), 1)

    prob_mask = cv2.remap(prob_mask, map_1, map_2, cv2.INTER_NEAREST)
    prob_mask = prob_mask * (1. - weight)

    for i in range(prob_mask.shape[-1]):
        prob_mask[:, :, i] += np.where(instance_im == i, 1., 0.) * weight

    return prob_mask


def get_colors():
    colors = np.zeros((17, 3))
    for k in range(17):
        if k == 0:
            colors[k, :] = np.zeros((3,))
        else:
            np.random.seed(k)
            rgb = np.random.randint(255, size=(1, 3)) / 255.0
            colors[k, :] = rgb

    return colors


def visualize_mask(prob_mask):
    image = np.zeros((prob_mask.shape[0], prob_mask.shape[1], 3))
    colors = get_colors()

    for i in range(prob_mask.shape[2]):
        image += np.expand_dims(prob_mask[:, :, i], -1) * np.expand_dims(np.expand_dims(colors[i], 0), 0)

    return image


if __name__ == '__main__':
    previous_color = cv2.imread("content/kitti_dataset/dataset/sequences/08/image_2/{:06d}.png".format(0),
                                cv2.IMREAD_UNCHANGED)
    rescale_factor = 5
    prob_mask = np.zeros((int(previous_color.shape[0] / rescale_factor), int(previous_color.shape[1] / rescale_factor),
                          17))
    prob_mask_no_flow = prob_mask.copy()
    print(previous_color.shape)

    # plt.ion()
    # f, ax = plt.subplots(2, 1)
    # plt.show()

    for i in range(1580, 1850):
        print("Frame {}".format(i+1))
        next_color = cv2.imread("content/kitti_dataset/dataset/sequences/08/image_2/{:06d}.png".format(i),
                                cv2.IMREAD_UNCHANGED)

        previous_gray = cv2.cvtColor(previous_color, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_color, cv2.COLOR_BGR2GRAY)

        instance_im = cv2.imread("content/kitti_dataset/dataset/sequences/08/instances_2/L{:06d}.png".format(i),
                                 cv2.IMREAD_UNCHANGED)
        instance_im = cv2.resize(instance_im, (int(instance_im.shape[1] / rescale_factor),
                                               int(instance_im.shape[0] / rescale_factor)),
                                 cv2.INTER_NEAREST)

        flow = calculate_flow(previous_gray, next_gray)
        flow = cv2.resize(flow, (int(flow.shape[1] / rescale_factor),
                                 int(flow.shape[0] / rescale_factor)),
                                 cv2.INTER_LINEAR) / rescale_factor

        previous_color = next_color
        prob_mask = update_mask(prob_mask, instance_im, flow, 0.5)
        # fake_flow = np.zeros_like(flow)
        # fake_flow[:, :, 0] = 30.
        # prob_mask_no_flow = update_mask(prob_mask_no_flow, instance_im, fake_flow, 0.5)

        vis_im = visualize_mask(prob_mask)
        cv2.imwrite("results/{}.png".format(i), vis_im * 255.)
        # ax[0].imshow(next_color)
        # ax[1].imshow(instance_im)
        # ax[1].imshow(visualize_mask(prob_mask))
        # ax[3].imshow(visualize_mask(prob_mask_no_flow))

        # plt.pause(0.001)


    exit()


