import pickle
import cv2
import numpy as np

"""
在给定相机标定好当前位置对应的外参以后，通过虚拟世界坐标系上的3个点进行定位，计算从图像坐标系(x_i, y_i)反向投影回世界坐标系(x_w, y_w, 0)所需要的的各项参数
该脚本执行需要有指定的相机参数pkl文件
最终结果为一个vec_uvs，用于从(u, v)计算(x_c, y_c, z_c)，和一个mat_trans_inv，用于计算最终坐标
两个参数会保存为re_project_param.pkl文件
"""

file = open('./calibration/parameters_in.pkl', 'rb')
parameters = pickle.load(file)
file.close()

camera_matrix = parameters['camera_matrix']
dist_coeff = parameters['dist_coeff']
rvecs = parameters['rvecs'][-1]
tvecs = parameters['tvecs'][-1]

grid_length = 5  # 棋盘格一个格子的边长（5mm）
row_count = 28

# 找三个points_world
points_world = np.zeros((3, 3), np.float32)
points_world[0, :2] = [0., 0.]
points_world[1, :2] = [0., (row_count-1)*grid_length]
points_world[2, :2] = [(row_count-1)*grid_length, 0.]
points_world_h = np.squeeze(cv2.convertPointsToHomogeneous(points_world)).T

# 计算mat_trans = A(R|t)，通过mat_trans从points_world计算points_c
mat_r = cv2.Rodrigues(rvecs)
mat_rt = np.zeros((3, 4))
mat_rt[:, :3] = mat_r[0]
mat_rt[:, -1] = np.squeeze(tvecs)

mat_trans = np.matmul(camera_matrix, mat_rt)

points_c = np.matmul(mat_trans, points_world_h)

# 根据points_c中的三个点，求解平面参数z_c = k1*x_c+k2*y_c+b
points_c_z = points_c[-1, :]
points_c_xy_h = np.ones((3, 3), np.float32)
points_c_xy_h[:, :2] = points_c[:2, :].T
vec_uvs = np.matmul(np.linalg.inv(points_c_xy_h), points_c_z)

# 求mat_rt的伪逆，作为反变换矩阵，从points_uv计算points_uvs，最后反变换为points_world_eval
mat_trans_inv = np.linalg.inv(mat_trans[:, (0, 1, 3)]) # np.linalg.pinv(mat_trans) # 

re_project_param = {'vec_uvs': vec_uvs, 'mat_trans_inv': mat_trans_inv}

output = open('./calibration/re_project_param.pkl', 'wb')
pickle.dump(re_project_param, output)
output.close()