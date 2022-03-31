import cv2
import argparse
import sys
import os
import os.path as osp

from utils.loggers import Logger
import utils.tool as tool
from detectors import Parallel, Skeleton, Hybrid
import configs


def main():
    """Main"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str,
                        default='', help='path to config file')
    parser.add_argument('--file', type=str, default='1.png',
                        help='path to image file for width detection')
    args = parser.parse_args()

    sys.stdout = Logger()

    cfg = configs.init_config()
    save_dir = sys.stdout.get_save_dir()
    print('test detector')

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    tool.set_random_seed(0)

    if args.file:
        img = cv2.imread(args.file)
    else:
        img = cv2.imread('test.png')

    cfg.detector.img_size_x = img.shape[1]
    cfg.detector.img_size_y = img.shape[0]
    cfg.freeze()
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

    img, binary, contours = detector.preprocess(img)
    lines, patches = detector.run(binary, contours)

    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

    cv2.imwrite(osp.join(save_dir, 'res.png'), img)
    print('An amount of %d lines are detected for width measuring.' % len(lines))


if __name__ == '__main__':
    main()
