import numpy as np


# TODO: Implement functions here:

# Obtains the depth image from a disparity image.
def disparity_to_depth(disparity_image, f_x, baseline, min_disparity=7600):
    # Adjust min and max disparity here!
    disparity_image = disparity_image.astype(float)
    disparity_image[np.logical_or(disparity_image == 65535, disparity_image < min_disparity)] = np.nan

    # TODO: Check this conversion Johannes!
    return baseline * f_x / ((disparity_image + 0.5) / 65536. * 49.)


# Projects a screen pixel to world coordinates.
def project_world(pixel_pos, depth, transform, intrinsic_matrix, img_shape):
    return transform.dot(project_3d(pixel_pos, depth, intrinsic_matrix, img_shape))


# Projects a pixel into 3D coordinates (relative to camera)
def project_3d(pixel_pos, depth, intrinsic_matrix, img_shape):
    print("TODO: Implement")


# Calculates the difference between two coordinates in meters (x to east, y to north).
# See below for example.
def coord_diff_to_metric_diff(d_lat, d_long, lat):
  d_y = d_lat * 110574
  d_x = d_long * 111320 * np.cos(np.radians(lat))
  return d_x, d_y


# Converts the metric difference to differences in latitude and longitude.
# See below for example.
def metric_diff_to_coord_diff(d_x, d_y, lat):
  d_lat = d_y / 110574
  d_long = d_x / 111320 / np.cos(np.radians(lat))
  return d_lat, d_long


# Returns the transformation matrix: world frame (-> vehicle frame) -> camera frame
# Usage: [x_world; 1] = camera_frame_to_world_transform(args).dot([x_camera_frame; 1])
# All angles in radians, distances in meters!
# See below for example.
def camera_frame_to_world_transform(heading, yaw_ext, pitch_ext, roll_ext, x_ext, y_ext, z_ext):
    # Coordinate axis switch for camera.
    C_c = np.array([[0., -1.,  0., 0.],
                    [0.,  0., -1., 0.],
                    [1.,  0.,  0., 0.],
                    [0.,  0.,  0., 1.]])
    # Vehicle to camera transformation matrix.
    c_y = np.cos(yaw_ext)
    c_p = np.cos(pitch_ext)
    c_r = np.cos(roll_ext)
    s_y = np.sin(yaw_ext)
    s_p = np.sin(pitch_ext)
    s_r = np.sin(roll_ext)
    T_v_c = np.array([[c_y*c_p, c_y*s_p*s_r-s_y*c_r, c_y*s_p*c_r+s_y*s_r, x_ext],
                      [s_y*c_p, s_y*s_p*s_r+c_y*c_r, s_y*s_p*c_r-c_y*s_r, y_ext],
                      [   -s_p,             c_p*s_r,             c_p*c_r, z_ext],
                      [     0.,                  0.,                  0.,    1.]])

    # World to vehicle transformation matrix.
    c_h = np.cos(heading)
    s_h = np.sin(heading)
    T_w_v = np.array([[c_h, -s_h, 0., 0.],
                      [s_h,  c_h, 0., 0.],
                      [ 0.,   0., 1., 0.],
                      [ 0.,   0., 0., 1.]])

    # Return camera space to vehicle transform.
    return T_w_v.dot(T_v_c.dot(np.linalg.inv(C_c)))


def pcl_to_image(pointcloud, T_pcl_center_to_cam, intrinsic_matrix, img_shape):
    """ Projects a pointcloud to camera image.
    int
    img_shape: Tuple containing (width, height)
    """
    image_width = img_shape[1]
    image_height = img_shape[0]

    depth_image = np.nan * np.zeros((image_height, image_width), dtype=np.float32)

    pcl_inside_view = pointcloud
    pcl_inside_view_xyz = np.hstack((pcl_inside_view,
                                    np.ones((pcl_inside_view.shape[0], 1))))

    pcl_inside_view_xyz = T_pcl_center_to_cam.dot(pcl_inside_view_xyz.T)
    pcl_inside_view_xyz = pcl_inside_view_xyz[:, pcl_inside_view_xyz[2, :] > 0]
    pcl_inside_view = pcl_inside_view_xyz[:3, :].T

    pcl_projected = np.array(intrinsic_matrix).dot(pcl_inside_view_xyz)
    pixel = np.rint(pcl_projected / pcl_projected[2, :]).astype(int)[:2, :]

    index_bool = np.logical_and(
      np.logical_and(0 <= pixel[0], pixel[0] < image_width),
      np.logical_and(0 <= pixel[1], pixel[1] < image_height))
    pixel = pixel[:, index_bool]

    pcl_inside_view = pcl_inside_view[index_bool, :]

    # Considering occlusion, we need to be careful with the order of assignment
    # descending order according to the z coordinate.
    index_sort = pcl_inside_view[:, 2].argsort()[::-1]
    pixel = pixel[:, index_sort]
    pcl_inside_view = pcl_inside_view[index_sort, :]

    depth_image[pixel[1], pixel[0]] = pcl_inside_view[:, 2]

    return depth_image