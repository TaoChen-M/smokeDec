import random
import numpy as np
import cv2


def re_project(img_points, camera_matrix, dist_coeff, vec_uvs, mat_trans_inv):
    """
    img_points: [n, 1, 2], points with opencv format

    return: points [n, 3]
    """
    points_uv = np.squeeze(cv2.undistortPoints(
        img_points, camera_matrix, dist_coeff, None, camera_matrix))
    points_num = points_uv.shape[0]

    points_uvs = np.zeros((3, points_num), np.float32)
    points_uvs[2, :] = vec_uvs[-1] / \
                       (1. - vec_uvs[0] * points_uv[:, 0] - vec_uvs[1] * points_uv[:, 1])
    points_uvs[:2, :] = points_uv.T * points_uvs[2, :]

    points_world_eval = np.matmul(mat_trans_inv, points_uvs)
    points_world_eval[2, :] = 0.
    points_world_eval = points_world_eval.T

    return points_world_eval


def make_trans_mat(camera_matrix, rvecs, tvecs):
    trans_mat = np.zeros((3, 4))
    trans_mat[:, :3] = cv2.Rodrigues(rvecs)[0]
    trans_mat[:, -1] = np.squeeze(tvecs)
    trans_mat = np.matmul(camera_matrix, trans_mat)

    return trans_mat


def load_camera_parameters(file_str):
    import pickle
    file = open(file_str, 'rb')
    parameters = pickle.load(file)
    file.close()

    camera_matrix = parameters['camera_matrix']
    dist_coeff = parameters['dist_coeff']
    rvecs = parameters['rvecs'][-1]
    tvecs = parameters['tvecs'][-1]

    return camera_matrix, dist_coeff, rvecs, tvecs


def dist_points(pts1, pts2):
    """
    pts1: [num, 2/3] coordinates
    pts2: [num, 2/3] coordinates
    """

    if pts1.shape[1] == 2:
        dists = ((pts1[:, 0] - pts2[:, 0]) ** 2 + (pts1[:, 1] - pts2[:, 1]) ** 2) ** 0.5
    else:
        dists = ((pts1[:, 0] - pts2[:, 0]) ** 2 + (pts1[:, 1] - pts2[:, 1])
                 ** 2 + (pts1[:, 2] - pts2[:, 2]) ** 2) ** 0.5

    mean_dist = np.mean(dists)

    return dists, mean_dist


def dist(re):
    if re is None:
        return None
    return dist_sq(re) ** 0.5


def dist_sq(re):
    if re is None:
        return None
    return (re[2] - re[0]) ** 2 + (re[3] - re[1]) ** 2


def find_width(center_point, img, edge_value=10, padding=30):
    """Find the other point and calculate width"""
    # top, bottom = max(center_point[1] - padding, 0), min(center_point[1] + padding, img.shape[0])
    top, bottom = center_point[1] - padding, center_point[1] + padding
    if top >= bottom:
        return None, None

    # left, right = max(center_point[0] - padding, 0), min(center_point[0] + padding, img.shape[1])
    left, right = center_point[0] - padding, center_point[0] + padding
    if left >= right:
        return None, None

    h, w = img.shape[:2]
    img = img[top: bottom, left: right]

    # calculate the position of each point in the cropped image
    cropped_center_point = (center_point[0] - left, center_point[1] - top)
    edge_points = list(zip(*(np.where(img == edge_value)[::-1])))

    patches, dists = generate_lines(cropped_center_point, edge_points, img.shape)

    indexes = filter_lines_by_skeleton(patches, img)

    if len(indexes) == 0:
        return None, None

    idx = indexes[0]
    dist = dists[idx]
    for i in indexes:
        if dists[i] < dist:
            idx = i
            dist = dists[i]

    point_1, point_2 = cropped_center_point, edge_points[idx]

    shift = (round((point_2[0] - point_1[0]) / 2), round((point_2[1] - point_1[1]) / 2))

    patch_top, patch_bottom = top + shift[1], bottom + shift[1]
    patch_left, patch_right = left + shift[0], right + shift[0]
    point_1 = (point_1[0]-shift[0], point_1[1]-shift[1])
    point_2 = (point_2[0]-shift[0], point_2[1]-shift[1])

    if patch_top < 0 or patch_left < 0 or patch_bottom > h or patch_right > w or \
            patch_bottom-patch_top != 2*padding or patch_right-patch_left != 2*padding:
        raise ValueError

    return (point_1[0], point_1[1], point_2[0], point_2[1]), \
           (patch_top, patch_bottom, patch_left, patch_right)


def filter_lines_by_skeleton(patch_images, img):
    """Filter lines by skeleton"""
    return [idx for idx, patch in enumerate(patch_images) if np.any(patch == img)]


def generate_lines(start_point, end_points, img_shape):
    """Generate line"""
    start_point = tuple(start_point)
    img_shape = tuple(img_shape)

    patch_images = []
    dists = []

    for point_x, point_y in end_points:
        patch_img = np.zeros(img_shape, dtype=np.uint8) + 64
        patch_img = cv2.line(patch_img, start_point,
                             (point_x, point_y), (0, 0, 0), 1)

        patch_images.append(patch_img)
        dists.append(
            np.sqrt((start_point[0] - point_x) ** 2 + (start_point[1] - point_y) ** 2))

    return patch_images, dists


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_patch_with_line_rad(img, line, rad):
    top = min(line[1], line[3]) - rad
    bottom = max(line[1], line[3]) + rad + 1
    left = min(line[0], line[2]) - rad
    right = max(line[0], line[2]) + rad + 1

    if len(img.shape) > 2:
        patch = img[top: bottom, left: right, :]
    else:
        patch = img[top: bottom, left: right]

    return patch


def get_patch_with_center_rad(img, center, rad):
    top, bottom = max(center[1] - rad, 0), min(center[1] + rad + 1, img.shape[0])
    if top >= bottom:
        return None, None

    left, right = max(center[0] - rad, 0), min(center[0] + rad + 1, img.shape[1])
    if left >= right:
        return None, None

    if len(img.shape) > 2:
        patch = img[top: bottom, left: right, :]
    else:
        patch = img[top: bottom, left: right]

    return patch
