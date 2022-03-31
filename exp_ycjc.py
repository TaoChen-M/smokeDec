"""
完整的烟叶宽度检测算法脚本，从图片读取->调用detector进行位置检测，根据检测结果产生patches->由深度模型进行筛选->得到筛选后结果->利用reproject函数计算实际宽度
->形成统计结果
"""

from PIL.Image import Image
import cv2
import argparse
import os
import sys
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

from utils.loggers import Logger
from utils import tool
from detectors import Parallel, Skeleton, Hybrid
from classifier import Model
import configs
from torchvision import transforms as T
from torch.autograd import Variable

def main(source_img):
    """Main"""
    # initialization of parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str,
                        default='', help='path to config file')
    parser.add_argument('--file', type=str, default='',
                        help='path to image file for width detection')
    parser.add_argument('--model', default='resnet50',
                        choices=['resnet18', 'resnet34',
                                 'resnet50', 'resnet101', 'resnet152'],
                        help='model to classify patches')
    parser.add_argument('--model-path', default='91.pth',
                        help='path to load model')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--camera-param', default='calibration/parameters_in.pkl',
                        help='path to the camara parameter file in calibration(.pkl)')
    parser.add_argument('--re-projection-param',
                        default='calibration/re_project_param.pkl', help='path to re_project_param.pkl')
    args = parser.parse_args()

    # initialization of logger and save dir. After the pipeline, a record file (record.log), 3 images (image.png, res_pre_filter.png, res.png) will be saved into the save dir. The image of statistic result from matplotlib is also saved here.
    sys.stdout = Logger()
    save_dir = sys.stdout.get_save_dir()
    print('test the whole pipeline from detection->filter->re-projection->statistic results.')

    cfg = configs.init_config()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    tool.set_random_seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # read image. executed after capture image by camera.
    # if args.file:
    #     # img = cv2.imread(args.file)
    #     img = cv2.imread(source_img)
    # else:
    #     img = cv2.imread('test.png')


    img = cv2.imread(source_img)
    # save the first image
    cv2.imwrite(os.path.join(save_dir, 'image.png'), img)
    cfg.detector.img_size_x = img.shape[1]
    cfg.detector.img_size_y = img.shape[0]
    # cfg.freeze()
    print(cfg)

    if cfg.detector.method == 'parallel':
        detector = Parallel()
    elif cfg.detector.method == 'skeleton':
        detector = Skeleton()
    elif cfg.detector.method == 'hybrid':
        detector = Hybrid()
    elif cfg.detector.method == 'duo':
        detector = Parallel()
        detector_1 = Skeleton()
    else:
        return

    # detection
    img, binary, contours = detector.preprocess(img)
    lines, patches = detector.run(binary, contours)

    # Draw all the lines on the image, and save the image
    img_copy = img.copy()  
    for line in lines:
        cv2.line(img_copy, (line[0], line[1]),
                 (line[2], line[3]), (0, 0, 255), 1)

    # save the second image
    cv2.imwrite(os.path.join(save_dir, 'res_pre_filter.png'), img_copy)
    print('An amount of %d lines are detected for width measuring.' % len(lines))

    # generate patches for deep model
    patch_images = []
    for idx in range(len(lines)):
        curr_line, curr_patch = lines[idx], patches[idx]
        patch_image = img[curr_patch[0]:curr_patch[1],
                          curr_patch[2]:curr_patch[3]].copy()
        cv2.line(patch_image, (curr_line[0] - curr_patch[2], curr_line[1] - curr_patch[0]),
                 (curr_line[2] - curr_patch[2], curr_line[3] - curr_patch[0]), (0, 0, 255), 1)
        patch_image = cv2.cvtColor(patch_image, cv2.COLOR_BGR2RGB)
        patch_images.append(patch_image)

    # Initiate classifier and predict labels
    model = Model(args.model)
    if torch.cuda.is_available():
        model.backbone.cuda()
    model.load(args.model_path)


    outputs = []
    for start in range(0, len(patch_images), args.batch_size):
        curr_batch = patch_images[start: start + args.batch_size]
        outputs += model.predict(curr_batch)
    outputs = torch.tensor(outputs)

    # outputs = [model.predict(patch_image) for patch_image in patch_images]
    # outputs = torch.cat(outputs)

    labels = torch.argmax(outputs, dim=1).tolist()

    patch_dir = os.path.join(save_dir, 'patches')
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    for idx in range(len(lines)):
        cv2.imwrite(os.path.join(patch_dir, str(idx) + '_' + str(labels[idx]) + '.png'),
                    cv2.cvtColor(patch_images[idx], cv2.COLOR_RGB2BGR))

    # save the filtered results as the third image
    lines_filtered = [lines[idx] for idx in range(len(lines)) if labels[idx]]

    for line in lines_filtered:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)

    cv2.imwrite(os.path.join(save_dir, 'res.png'), img)
    print('An amount of %d lines pass the filter as final resutls.' %
          len(lines_filtered))

    # re-projection
    file = open(args.camera_param, 'rb')
    parameters = pickle.load(file)
    file.close()
    camera_matrix = parameters['camera_matrix']
    dist_coeff = parameters['dist_coeff']

    file = open(args.re_projection_param, 'rb')
    re_project_param = pickle.load(file)
    file.close()
    vec_uvs = re_project_param['vec_uvs']
    mat_trans_inv = re_project_param['mat_trans_inv']

    line_points_start = np.empty((len(lines_filtered), 1, 2), dtype=np.float32)
    for idx in range(len(lines_filtered)):
        line_points_start[idx, 0, :] = lines_filtered[idx][:2]

    points_world_start = tool.re_project(
        line_points_start, camera_matrix, dist_coeff, vec_uvs, mat_trans_inv)

    line_points_end = np.empty((len(lines_filtered), 1, 2), dtype=np.float32)
    for idx in range(len(lines_filtered)):
        line_points_end[idx, 0, :] = lines_filtered[idx][2:]

    points_world_end = tool.re_project(
        line_points_end, camera_matrix, dist_coeff, vec_uvs, mat_trans_inv)

    dists_world, _ = tool.dist_points(points_world_start, points_world_end)

    # save histogram into save dir
    bins = np.arange(0.2, 2.0, 0.1)
    plt.hist(dists_world, bins)
    plt.xlabel('mm')
    plt.savefig('hist.png')


if __name__ == '__main__':
    # for file in os.listdir('imgs'):
    #     print("test file:{}".format(file))
    #     main('imgs/'+file)
    main('6/save_1161718192021_+6.bmp')
