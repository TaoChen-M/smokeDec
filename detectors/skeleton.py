import detectors
import numpy as np
import cv2
import utils.tool as util


class Skeleton(detectors.Detector):

    def __init__(self):
        r"""parameter initialization

        """
        super(Skeleton, self).__init__()

    def run(self, binary, contours):
        # 二进制图片反转
        binary = 255-binary

        # labels表示不同数字表示的连通域
        num_labels, labels, stats, _ = \
            cv2.connectedComponentsWithStats(
                binary, connectivity=8, ltype=cv2.CV_32S)

        # 创建一张白色底板，并将各个连通域置为灰色
        mask = np.zeros(binary.shape, dtype=np.uint8) + 255
        mask[labels != 0] = 128

        # 提取区域中心线，mask上置为黑色
        skeleton = cv2.ximgproc.thinning(binary)

        mask[skeleton == 255] = 0

        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 直接使用preprocess的结果
        cv2.drawContours(mask, contours, -1, (10, 10, 10), 1)

        # 提取所有的轮廓点，并进行排序
        indexes = list(zip(*np.where(mask == 10)))
        np.random.shuffle(indexes)

        mask_com = np.zeros(mask.shape, dtype=np.uint8) + 255

        lines = []
        patches = []

        for y, x in indexes:
            line, patch = util.find_width((x, y), mask, 10)
            if line is None:
                continue
            dist = ((line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2) ** 0.5

            line_2, patch_2 = util.find_width(
                (line[2] + patch[2], line[3] + patch[0]), mask, 10)
            if line_2 is not None:
                dist_2 = ((line_2[0] - line_2[2]) ** 2 +
                          (line_2[1] - line_2[3]) ** 2) ** 0.5
                if dist > dist_2:
                    line, patch, dist = line_2, patch_2, dist_2

            top, bottom, left, right = patch
            line = [line[0] + left, line[1] + top,
                    line[2] + left, line[3] + top]

            if dist > 30 or mask_com[line[1], line[0]] == 0 or mask_com[line[3], line[2]] == 0:
                continue

            mask_com[top: bottom, left: right] = 0

            print(len(lines))
            lines.append(line)
            patches.append(patch)

            if len(lines) >= self.cfg.skeleton.max_res:
                return lines, patches

        return lines, patches

    # def find_width_out(self, p, mask, method):
    #     if method == 'once':
    #         re, dist = self.find_width(p, mask)
    #         return re, dist
    #
    #     if method == 'twice':
    #         re1, dist1 = self.find_width(p, mask)
    #         if re1 is None:
    #             return re1, dist1
    #
    #         p2 = [re1[2], re1[3]]
    #         re2, dist2 = self.find_width(p2, mask)
    #
    #         if re2 is None:
    #             return re1, dist1
    #
    #         if dist2 > dist1:
    #             return re1, dist1
    #         else:
    #             return re2, dist2
    #
    #     if method == 'iter':
    #         re, dist = self.find_width(p, mask)
    #         if re is None:
    #             return re, dist
    #
    #         # i = 1
    #         while True:
    #             re_n, dist_n = self.find_width([re[2], re[3]], mask)
    #
    #             if re_n is None or dist_n >= dist:
    #                 return re, dist
    #
    #             re = re_n
    #             dist = dist_n
    #             # print(i)
    #             # i = i+1
    #
    #     return None, None
