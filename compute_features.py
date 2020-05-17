import numpy as np
import os
import cv2
import pykitti
import json
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from skimage.measure import LineModelND, ransac
from operator import itemgetter

from tools import utils

QUANTILE = True
FIT_LINE = False
FIT_BOX = True
MERGE_BBOXES = 1

# Choose number of filter for mergedbbox
mbboxnr = 2

# Choose error for merging bboxes
inflation = 0.1

num_filt = 3


def isOverlapping1D(xmin1, xmin2, xmax1, xmax2) :
    xmin1 -= inflation
    xmin2 -= inflation
    xmax1 += inflation
    xmax2 += inflation

    if xmax1 >= xmin2 and xmax2 >= xmin1:
        return True
    else:
        return False


def isOverlapping3D(box1, box2):
    if isOverlapping1D(box1[0], box2[0], box1[3], box2[3]) and isOverlapping1D(box1[1], box2[1], box1[4], box2[4]) and \
            isOverlapping1D(box1[2], box2[2], box1[5], box2[5]):
        return True
    else:
        return False


def quantile_filt(u, v, z):
    """removes points that exceed the 40% quantile in z direction
    """
    #upper limit for z
    if u.ndim > 1:
        u_p = u[:,0]
        v_p = v[:,0]
        z_p = z[:,0]
    else:
        u_p, v_p, z_p = u, v, z
    upperlim = np.quantile(z, 0.2)

    #cut of all z that are bigger than upperlim
    z_m = np.where(z_p > upperlim, np.nan, z_p)
    u_m = np.where(z_p > upperlim, np.nan, u_p)
    v_m = np.where(z_p > upperlim, np.nan, v_p)
    u, v, z = expand_concat(u, v, z, u_m, v_m, z_m)
    return u, v, z


def fit_line(pcl, class_id):
    """ fits a line to pointclouds, that are labeled as pole"""
    # camera coordinates
    x, y, z = pcl
    if class_id == 10:
        # Get indices of data
        indices = list(range(x.shape[0]))
        # Define minimum number of iterations
        N = int(len(x)*0.5)
        d = len(x) *0.2
        error_opt = np.Inf
        inlier_opt = np.array([False]*len(x))
        # Subsample minimum number of datapoints to create model 
        subsets = np.random.choice(indices, N)
        for subset in subsets:
            # Compute errors
            x_error = abs(x - x[subset])
            z_error = abs(z - z[subset])
            # Define error thresholds
            x_tresh = 0.2
            z_thresh = 0.4

            inlier_mask = np.logical_and(x_error < x_tresh, z_error < z_thresh)
            if inlier_mask.sum() > d:
                x_model = x[inlier_mask].mean()
                z_model = z[inlier_mask].mean()
                    
                x_error = abs(x_model - x[inlier_mask])
                z_error = abs(z_model - z[inlier_mask])
                error = (x_error**2 + z_error**2).mean()**0.5
                if error < error_opt:
                    error_opt = error
                    inlier_opt = inlier_mask
        x = np.where(inlier_opt, x, np.nan)
        y = np.where(inlier_opt, y, np.nan)
        z = np.where(inlier_opt, z, np.nan)
        
        return x, y, z
    else:
        return x, y, z


def fit_box(pcl, class_id):
    """ fits a line to pointclouds, that are labeled as pole"""
    # camera coordinates
    x, y, z = pcl
    if class_id == 10:
       return fit_line(pcl, class_id)
    # Get indices of data
    indices = list(range(x.shape[0]))
    # Define minimum number of iterations N
    # N = int(math.log(1-p)/math.log(1-(1-e)**2))
    N = int(0.5 * len(x))
    max_iter = 20
    # Define error thresholds
    e = 0.15
    density_opt = 0
    inlier_opt = np.array([False]*len(x))
    # Subsample minimum number of datapoints to create model 

    subsets = np.random.choice(indices, N)

    for subset in subsets:
        d = 1
        density = [0]
        # Compute bbox
        box_min = pcl[:,subset].reshape(3,-1)
        box_max = pcl[:,subset].reshape(3,-1)
        # print(box_min.shape)
        box_min = box_min - e
        box_max = box_max + e
        # Compute inliers
        inlier_mask = np.logical_and(np.all(pcl >= box_min, axis=0), np.all(pcl <= box_max, axis=0))
        # iteratively extend box size
        i = 0
        while inlier_mask.sum() > d and i < max_iter:
            box_min = np.min(pcl[:,inlier_mask], axis=1).reshape(3,-1)
            box_max = np.max(pcl[:,inlier_mask], axis=1).reshape(3,-1)
            vol = (box_max - box_min).prod()
            d = inlier_mask.sum()
            if d > len(x) * 0.2 and vol > 0:
                density.append(d/vol)
                inlier_sample = inlier_mask
            # Increase box size
            box_min = box_min - e
            box_max = box_max + e
            inlier_mask = np.logical_and(np.all(pcl > box_min, axis=0), np.all(pcl < box_max, axis=0))
            i += 1

        if max(density) > density_opt:
            density_opt = max(density)
            inlier_opt = inlier_sample

    x = np.where(inlier_opt, x, np.nan)
    y = np.where(inlier_opt, y, np.nan)
    z = np.where(inlier_opt, z, np.nan)
    return x, y, z

def merge_pole(pcls, bboxs, classes_list):
    error_dist = 2
    bboxs = bboxs[:,:,mbboxnr]
    x = []
    y = []
    z_min = []
    z_max = []
    for i, pcl in enumerate(pcls):
        pcl = pcl[:,:,mbboxnr]
        pcl = pcl[~np.isnan(pcl)].reshape(4, -1)
        if pcl.size > 0:
            x_t, y_t, _, _ = pcl.mean(axis=1)
            _, _, z_min_t, _ = pcl.min(axis=1)
            _, _, z_max_t, _ = pcl.max(axis=1)
            x.append(x_t)
            y.append(y_t)
            z_min.append(z_min_t)
            z_max.append(z_max_t)
    x = np.asarray(x)
    y = np.asarray(y)
    z_min = np.asarray(z_min).reshape(-1, 1)
    z_max = np.asarray(z_max).reshape(-1, 1)
    xx = np.meshgrid(x, x)
    yy = np.meshgrid(y, y)
    dist = ((xx[0]-xx[1])**2 + (yy[0]-yy[1])**2)**0.5
    close = np.where(dist<error_dist, True, False)
    indices = np.argsort(close.sum(axis=0))[::-1]
    x_mean = np.zeros(x.shape)
    y_mean = np.zeros(x.shape)
    z_min_m = np.zeros(x.shape)
    z_max_m = np.zeros(x.shape)
    bboxs_merged = []
    class_list_out = []
    merged = np.zeros(x.shape, dtype=bool)
    for i in indices:
        neighbors = close[i]
        if not merged[i] and sum(neighbors) > 1 and classes_list[i]==10:
            x_mean = x[neighbors].mean()
            y_mean = y[neighbors].mean()
            z_min_m = z_min[neighbors].min()
            z_max_m = z_max[neighbors].max()
            bboxs_merged.append(np.array([x_mean - 0.1, y_mean - 0.1, z_min_m, x_mean + 0.1, y_mean + 0.1,  z_max_m]).reshape(1,6))
            class_list_out.append(classes_list[i])
            merged = np.logical_or(merged, neighbors)
        elif not merged[i]:
            bboxs_merged.append(bboxs[i].reshape(1,6))
            merged = np.logical_or(merged, neighbors)
            class_list_out.append(classes_list[i])
    return bboxs_merged, class_list_out    


def expand_concat(u, v, z, u_m, v_m, z_m):
    if u.ndim == 1:    
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
        pcl = point_cloud[0:3, :, m][~np.isnan(point_cloud[0:3, :, m])]
        if pcl.size > 0:
            min_coord = np.min(point_cloud[0:3, :, m][~np.isnan(point_cloud[0:3, :, m])].reshape(3, -1, 1), axis=1)
            max_coord = np.max(point_cloud[0:3, :, m][~np.isnan(point_cloud[0:3, :, m])].reshape(3, -1, 1), axis=1)
            bbox[:, m] = np.array([min_coord, max_coord]).reshape((6,))
        else:
            bbox[:, m] = np.array([np.nan]*6).reshape((6,))
    return bbox


if __name__ == '__main__':
    basedir = 'content/kitti_dataset/dataset'
    sequence = '08'

    dataset = pykitti.odometry(basedir, sequence)
    baseline = dataset.calib.b_rgb
    P_cam2 = dataset.calib.P_rect_20
    f_x = P_cam2[0, 0]
    f_y = P_cam2[1, 1]
    u_0 = P_cam2[0, 2]
    v_0 = P_cam2[1, 2]
    print(P_cam2)

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

    # sorting alldata for image_id
    all_data = all_data['results']
    all_data_sort = []
    for x in sorted(all_data, key = itemgetter('image_id')):
        all_data_sort.append(x)

    path = "content/kitti_dataset/dataset/sequences/08"
    out_path = "results"

    results = []
    # store bboxes and pcls
    
    if MERGE_BBOXES:
        bboxes = []
        pcls = []
        transforms = []
        classes_list = []

    for n, data in enumerate(all_data_sort):
        Only processing half of the images
        if n % 2 == 0:
           print("Skip processing frame " + data['image_id'] + '.')
           continue

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

            # Continue if there are less then 5 points per object
            if z.size < 5:
                continue
            
            if QUANTILE:
                u, v, z = quantile_filt(u, v, z)

            if FIT_LINE:
                u, v, z = expand_concat(u, v, z, u[:,1], v[:,1], z[:,1])

            if FIT_BOX:
                u, v, z = expand_concat(u, v, z, u[:,1], v[:,1], z[:,1])

            point_cloud = np.zeros((4, np.size(z,0), num_filt))
            for j in range(z.shape[0]):
                point_cloud[0, j, :] = z[j,:] * (u[j,:] - u_0) / f_x
                point_cloud[1, j, :] = z[j,:] * (v[j,:] - v_0) / f_y
                point_cloud[2, j, :] = z[j,:] 
                point_cloud[3, j, :] = 1.0

            if FIT_LINE:
                point_cloud[0:3, :, -2] = fit_line(point_cloud[0:3,:, 0], class_ids[i])
            if FIT_BOX:
                point_cloud[0:3, :, -1] = fit_box(point_cloud[0:3,:, 0], class_ids[i])

            point_cloud = point_cloud.reshape((4, -1))
            transform = T_w0_w.dot(dataset.poses[frame_id].dot(T_cam0_cam2))

            # Ground truth poses are T_w_cam0
            point_cloud = transform.dot(point_cloud)
            point_cloud = point_cloud.reshape((4, -1, num_filt))

            bbox = get_bbox(point_cloud)

            if MERGE_BBOXES:
                pcls.append(point_cloud)
                transforms.append(transform)
                classes_list.append(class_ids[i])
                bboxes.append(bbox)
        
            # Save pcls and bboxes
            pcl_path = os.path.join(out_path, "landmark_f{}_i{}".format(data['image_id'], i))
            np.save(pcl_path, point_cloud, allow_pickle=False)
            bbox_path = os.path.join(out_path, "bbox_f{}_i{}".format(data['image_id'], i))
            np.save(bbox_path, bbox, allow_pickle=False)
            results.append([frame_id, i, class_ids[i], pcl_path, bbox_path, transform[:3, 3]])




    
    
    if MERGE_BBOXES==1:
        npbboxes = np.asarray(bboxes)
        mergedbboxes, class_list_out = merge_pole(pcls, npbboxes, classes_list)
        print(len(mergedbboxes))
        # Save pcls and mergedbboxes
        mergedbbox_path = os.path.join(out_path, "mergedbbox")
        np.save(mergedbbox_path, np.asarray(mergedbboxes), allow_pickle=False)
        classes_list_path = os.path.join(out_path, "classes_list")
        np.save(classes_list_path, np.array(class_list_out), allow_pickle=False)
        results.sort()
        with open(os.path.join(out_path, "_results.txt"), 'w') as f:
            for result in results:
                f.write(str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + " " + str(result[3]) + " " +
                        str(result[4]) + " " + str(result[5][0]) + " " + str(result[5][1]) + " " + str(result[5][2]) + "\n")
        print("Finished.")
    elif MERGE_BBOXES==2:
        if False:
            npbboxes = np.asarray(bboxes)

            # Number of unmerged landmarks
            nlandm = len(pcls)

            # Adjaceny matrix of bbox, if bbox intersect that entry gets 1, otherwise 0
            index = np.zeros((nlandm, nlandm))

            # All merged bboxes
            mergedbboxes = []
            for i in range(nlandm):
                for j in range(nlandm):
                    if isOverlapping3D(npbboxes[i, :, mbboxnr], npbboxes[j, :, mbboxnr]) and classes_list[i] == classes_list[j]:
                        index[i, j] = 1

            # Find for all intersecting bboxes minimum and maximum
            blacklist = np.array((-1,))
            class_list_out = []
            for i in range(nlandm):
                if i in blacklist:
                    continue
                elif np.sum(index[i,:]) != 0:
                    minoverlappingbboxes = npbboxes[np.where(index[i,:]==1),0:3,mbboxnr]
                    maxoverlappingbboxes = npbboxes[np.where(index[i,:]==1),3:6, mbboxnr]

                    con = np.concatenate([np.min(minoverlappingbboxes, axis=1), np.max(maxoverlappingbboxes, axis=1)],
                                            axis=1)
                    mergedbboxes.append(con)
                    blacklist_add = np.where(index[i,:]==1)[0]
                    blacklist = np.unique(np.concatenate((blacklist,blacklist_add),0))

                    class_list_out.append(classes_list[i])
        else:
            npbboxes = np.asarray(bboxes)
            nppcls = np.asarray(pcls)
            for i, pcl in enumerate(pcls):
                if classes_list[i] == 11: # if it is a pole
                    merge_pole(pcls, npbboxes)



        print(len(mergedbboxes))
        # Save pcls and mergedbboxes
        mergedbbox_path = os.path.join(out_path, "mergedbbox")
        np.save(mergedbbox_path, np.array(mergedbboxes))
        classes_list_path = os.path.join(out_path, "classes_list")
        np.save(classes_list_path, np.array(class_list_out))

    with open(os.path.join(out_path, "_results.txt"), 'w') as f:
        for result in results:
            f.write(str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + " " + str(result[3]) + " " +
                    str(result[4]) + " " + str(result[5][0]) + " " + str(result[5][1]) + " " + str(result[5][2]) + "\n")

    print("Finished.")
