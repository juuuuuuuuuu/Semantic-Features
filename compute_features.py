import numpy as np
import os
import cv2
import pykitti
import json
import matplotlib.pyplot as plt
from tools import utils

QUANTILE = True
Mergebboxes = True

#Choose number of filter for mergedbbox
mbboxnr = 1

num_filt = 2

def isOverlapping1D(xmin1, xmin2, xmax1, xmax2) :
    if xmax1 >= xmin2 and xmax2 >= xmin1:
        return True
    else:
        return False

def isOverlapping3D(box1, box2):
    if isOverlapping1D(box1[0], box2[0], box1[3], box2[3]) and isOverlapping1D(box1[1], box2[1], box1[4], box2[4]) and isOverlapping1D(box1[2], box2[2], box1[5], box2[5]):
        return True
    else:
        return False

def quantile_filt(u, v, z):
    """removes points that exceed the 40% quantile in z direction
    """
    #upper limit for z
    upperlim = np.quantile(z, 0.2)

    #cut of all z that are bigger than upperlim
    z_m = np.where(z > upperlim, np.nan, z)
    u_m = np.where(z > upperlim, np.nan, u)
    v_m = np.where(z > upperlim, np.nan, v)
    u, v, z = expand_concat(u, v, z, u_m, v_m, z_m)
    return u, v, z

def expand_concat(u, v, z, u_m, v_m, z_m):
    u = np.expand_dims(u, axis=1)
    v = np.expand_dims(v, axis=1)
    z = np.expand_dims(z, axis=1)
    u_m = np.expand_dims(u_m, axis=1)
    v_m = np.expand_dims(v_m, axis=1)
    z_m = np.expand_dims(z_m, axis=1)
    u = np.concatenate((u, u_m), axis=1)
    v = np.concatenate((v, v_m), axis=1)
    z = np.concatenate((z, z_m), axis=1)
    return u, v, z

def get_bbox(pcl):
    bbox = np.zeros((6, num_filt))
    for m in range(num_filt):
        min_coord = np.min(point_cloud[0:3,:,m][~np.isnan(point_cloud[0:3,:,m])].reshape(3,-1, 1), axis=1)
        max_coord = np.max(point_cloud[0:3,:,m][~np.isnan(point_cloud[0:3,:,m])].reshape(3,-1, 1), axis=1)
        bbox[:,m] = np.array([min_coord, max_coord]).reshape((6,))
    return bbox
if __name__ == '__main__':
    basedir = 'content/kitti_dataset/dataset'
    sequence = '04'

    dataset = pykitti.odometry(basedir, sequence)
    baseline = dataset.calib.b_rgb
    P_cam2 = dataset.calib.P_rect_20
    f_x = P_cam2[0, 0]
    f_y = P_cam2[1, 1]
    u_0 = P_cam2[0, 2]
    v_0 = P_cam2[1, 2]

    # World is twisted so we need to transform.
    T_w0_w = np.array([[ 0., 0., 1., 0.],
                       [-1., 0., 0., 0.],
                       [ 0.,-1., 0., 0.],
                       [ 0., 0., 0., 1.]])
    # The poses are saved as T_w_cam0, for this reason we need a transform between cam0 and cam2.
    T_cam0_cam2 = np.linalg.inv(dataset.calib.T_cam0_velo).dot(dataset.calib.T_cam2_velo)
    P_cam2 = np.vstack([P_cam2, np.array([0, 0, 0, 1])])

    frame_count = len(dataset.poses)

    with open("results.json") as json_file:
        all_data = json.load(json_file)

    path = "content/kitti_dataset/dataset/sequences/04"
    out_path = "results"

    results = []

#store bboxes and pcls
    if Mergebboxes:
        bboxes = []
        pcls = []
        transforms = []
        classes_list = []

    for data in all_data['results']:
        frame_id = int(data['image_id'])
        print("Processing frame " + data['image_id'] + '.')

        path_to_mask = os.path.join(path, "instances_2", "L{}.png".format(data['image_id']))
        path_to_disp = os.path.join(path, "disparity_2", "{}raw.png".format(frame_id))

        mask_image = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        disp_image = cv2.imread(path_to_disp, cv2.IMREAD_UNCHANGED)

        depth_image_stereo = utils.disparity_to_depth(disp_image, f_x, baseline)
        depth_image_stereo = cv2.GaussianBlur(depth_image_stereo, (3, 3), 0)

        velo = dataset.get_velo(frame_id)
        depth_image = utils.pcl_to_image(velo[:, :3], dataset.calib.T_cam2_velo,
                                         P_cam2, (mask_image.shape[0], mask_image.shape[1]))
        depth_image[:100, :] = depth_image_stereo[:100, :]
        
        # Create structuring element for erosion
        kernel = np.ones((1, 1), np.uint8) 

        class_ids = data['classes']
        for i in range(len(class_ids)):
            mask_i = cv2.erode(np.array(mask_image == i + 1, dtype=np.uint8), kernel)
            mask = np.where(np.logical_and(mask_i, np.logical_not(np.isnan(depth_image))))

            u = []
            v = []
            z = []

            u = mask[1][:]
            v = mask[0][:]
            z = depth_image[v, u]

            #Continue if there are less then 5 points per object
            if z.size<5:
                continue
            
            if QUANTILE:
                u, v, z = quantile_filt(u, v, z)

            point_cloud = np.zeros((4, np.size(z,0), num_filt))
            for j in range(z.shape[0]):
                point_cloud[0, j, :] = z[j,:] * (u[j,:] - u_0) / f_x
                point_cloud[1, j, :] = z[j,:] * (v[j,:] - v_0) / f_y
                point_cloud[2, j, :] = z[j,:] 
                point_cloud[3, j, :] = 1.0

            point_cloud = point_cloud.reshape((4, -1))
            transform = T_w0_w.dot(dataset.poses[frame_id].dot(T_cam0_cam2))

            # Ground truth poses are T_w_cam0
            point_cloud = transform.dot(point_cloud)
            point_cloud = point_cloud.reshape((4, -1, num_filt))
            bbox = get_bbox(point_cloud)

            if Mergebboxes:
                bboxes.append(bbox)
                pcls.append(point_cloud)
                transforms.append(transform)
                classes_list.append(class_ids[i])
        
            #Save pcls and bboxes
            pcl_path = os.path.join(out_path, "landmark_f{}_i{}".format(data['image_id'], i))
            np.save(pcl_path, point_cloud, allow_pickle=False)
            bbox_path = os.path.join(out_path, "bbox_f{}_i{}".format(data['image_id'], i))
            np.save(bbox_path, bbox, allow_pickle=False)
            results.append([frame_id, i, class_ids[i], pcl_path, bbox_path, transform[:3, 3]])

    

    if Mergebboxes:
        npbboxes = np.asarray(bboxes)

        #Adjaceny matrix of bbox, if bbox intersect that entry gets 1, otherwise 0
        index = np.zeros((len(pcls), len(pcls)))

        #All merged bboxes
        mergedbboxes = np.zeros((len(pcls),6))

        for i in range(len(pcls)):
            for j in range(len(pcls)):
                if isOverlapping3D(npbboxes[i,:,mbboxnr],npbboxes[j,:,mbboxnr]) and classes_list[i] == classes_list[j]:
                    index[i,j] = 1

        # #Find for all intersecting bboxes minimum and maximum
        for i in range(len(pcls)):
            if sum(index[i,:]) != 0:
                minoverlappingbboxes = npbboxes[np.where(index[i,:]==1),0:3,mbboxnr]
                maxoverlappingbboxes = npbboxes[np.where(index[i,:]==1),3:6, mbboxnr]
                mergedbboxes[i][0:3] = np.min(minoverlappingbboxes, axis=1)
                mergedbboxes[i][3:6] = np.max(maxoverlappingbboxes, axis=1)
            else:
                mergedbboxes[i] = npbboxes[i,0:6,mbboxnr]
        
        #Save pcls and mergedbboxes 
        mergedbbox_path = os.path.join(out_path, "mergedbbox")
        np.save(mergedbbox_path, mergedbboxes, allow_pickle=False)
        classes_list_path = os.path.join(out_path, "classes_list")
        np.save(classes_list_path, classes_list, allow_pickle=False)


    results.sort()
    with open(os.path.join(out_path, "_results.txt"), 'w') as f:
        for result in results:
            f.write(str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + " " + str(result[3]) + " " +
                    str(result[4]) + " " + str(result[5][0]) + " " + str(result[5][1]) + " " + str(result[5][2]) + "\n")

    print("Finished.")



