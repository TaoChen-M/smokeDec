import pickle
import cv2
import numpy as np
from utils import tool as util

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes

file = open('./calibration/parameters_in.pkl', 'rb')
parameters = pickle.load(file)
file.close()

camera_matrix = parameters['camera_matrix']
dist_coeff = parameters['dist_coeff']
rvecs = parameters['rvecs'][-1]
tvecs = parameters['tvecs'][-1]

file = open('./calibration/img_points.pkl', 'rb')
img_points = pickle.load(file)
file.close()
img_points = img_points[-1]

file = open('./calibration/re_project_param.pkl', 'rb')
re_project_param = pickle.load(file)
file.close()
vec_uvs = re_project_param['vec_uvs']
mat_trans_inv = re_project_param['mat_trans_inv']

grid_length = 5  # 棋盘格一个格子的边长（5mm）
row_count = 28

points_world_eval = util.re_project(img_points, camera_matrix, dist_coeff, vec_uvs, mat_trans_inv)

world_points = np.zeros((row_count * row_count, 3), np.float32)
world_points[:, :2] = np.mgrid[0:row_count * grid_length:grid_length, 0:row_count * grid_length:grid_length].T.reshape(-1, 2)

errors = util.dist_points(world_points, points_world_eval)

errors = errors[0].reshape(28, 28)
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(errors, cmap=plt.cm.hot_r)
plt.show()

print('test')
# points_img_eval = cv2.projectPoints(points_world, rvecs, tvecs, camera_matrix, dist_coeff)
# points_img_eval = points_img_eval[0]
# points_uv = np.squeeze(cv2.undistortPoints(points_img_eval, camera_matrix, dist_coeff, None, camera_matrix)) #(3, 2)
# eval_points_high_res_all = {}

# for i in range(row_count-1):
#     for j in range(row_count-1):
#         world_points_high_res = np.zeros(((100*grid_length)*(100*grid_length), 3), np.float32)
#         world_points_high_res[:, :2] = np.mgrid[(0+i)*grid_length:(1+i)*grid_length:0.01, (0+j)*grid_length:(1+j)*grid_length:0.01].T.reshape(-1, 2)
        
#         eval_points_high_res = cv2.projectPoints(world_points_high_res, rvecs, tvecs, camera_matrix, dist_coeff)
        
#         eval_points_high_res_all[(i, j)] = np.concatenate((np.squeeze(eval_points_high_res[0]), world_points_high_res[:, :2]), 1)
        
# output = open('./calibration/eval_points_high_res_all.pkl', 'wb')
# pickle.dump(eval_points_high_res_all, output)
# output.close()

# file = open('./calibration/eval_points_high_res_all.pkl', 'rb')
# eval_points_high_res_all = pickle.load(file)
# file.close()

# idx = 0
# eval_chess_board_points = np.zeros(((row_count-1)*(row_count-1), 5)) # [min_x, max_x, min_y, max_y, idx]
# eval_points_high_res_list = []
# for key, value in eval_points_high_res_all.items():
#     eval_points_high_res_list.append(value)
#     eval_chess_board_points[idx, :] = [np.min(value[:, 0]), np.max(value[:, 0]), np.min(value[:, 1]), np.max(value[:, 1]), idx]
#     idx = idx+1



# fig = plt.figure()
# x = np.linspace(0, 250000, 250000, endpoint=False).astype(np.int32)
# ids = np.linspace(0, 250000, 500, endpoint=False).astype(np.int32)
# y = eval_points_high_res_list[0][x, 0]
# plt.plot(x, y,color="red",linewidth=1)
# plt.show()




# world_points_high_res = np.zeros(((row_count*100*grid_length)*(row_count*100*grid_length), 3), np.float32)
# world_points_high_res[:, :2] = np.mgrid[0:row_count*grid_length:0.01, 0:row_count*grid_length:0.01].T.reshape(-1, 2) 
# # world_points_high_res_h = cv2.convertPointsToHomogeneous(world_points_high_res)
# # camera_points_high_res = np.matmul(trans_mat, np.squeeze(world_points_high_res_h).T).T 

# eval_points_high_res = cv2.projectPoints(world_points_high_res, rvecs, tvecs, camera_matrix, dist_coeff)

# # zc_high_res = camera_points_high_res[:, 2]

# output = open('./calibration/_high_res.pkl', 'wb')
# pickle.dump(zc_high_res, output)
# output.close()


# print(test)

# world_points = np.zeros((row_count * row_count, 3), np.float32)  # 初始化棋盘格上角点的世界坐标（三维）
# world_points[:, :2] = np.mgrid[0:row_count * grid_length:grid_length,
#                           0:row_count * grid_length:grid_length].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y（注意间隔5mm）

# eval_points = cv2.projectPoints(world_points, rvecs, tvecs, camera_matrix, dist_coeff)
# eval_points = eval_points[0]

# test1 = eval_points.reshape(28, 28, 2)
# test2 = world_points.reshape(28, 28, 3)

# error_list, mean_error = util.dist_points(np.squeeze(eval_points), np.squeeze(img_points))
# err_map = error_list.reshape((28, 28))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# im = ax.imshow(err_map, cmap=plt.cm.hot_r)
# plt.show()

# uvs = cv2.undistortPoints(eval_points, camera_matrix, dist_coeff, None, camera_matrix) # eval_points->img_points
# uvs = np.squeeze(uvs)

# trans_mat = np.zeros((3, 4))
# trans_mat[:, :3] = cv2.Rodrigues(rvecs)[0]
# trans_mat[:, -1] = np.squeeze(tvecs)
# trans_mat = np.matmul(camera_matrix, trans_mat)

# world_points_h = cv2.convertPointsToHomogeneous(world_points)
# camera_points = np.matmul(trans_mat, np.squeeze(world_points_h).T).T
# camera_points[:, 0] = camera_points[:, 0]/camera_points[:, 2]
# camera_points[:, 1] = camera_points[:, 1]/camera_points[:, 2]

# error_list, mean_error = util.dist_points(np.squeeze(uvs), np.squeeze(camera_points[:, :2]))
# err_map = error_list.reshape((28, 28))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# im = ax.imshow(err_map, cmap=plt.cm.hot_r)
# plt.show()