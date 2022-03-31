import detectors
import numpy as np
import cv2


class Parallel(detectors.Detector):

    def __init__(self):
        r"""parameter initialization
        
        """
        super(Parallel, self).__init__()

    def run(self,  binary, contours):
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 1)

        res_line = []
        res_patch = []

        line_num = 1
        point_distance_list = []

        # all lines find by hough，(x1, y1, x2, y2)
        hough_line = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180,
                                     threshold=5, minLineLength=3, maxLineGap=2)

        # Get the slope of a line
        lines_list = hough_line.reshape(-1, 4)

        slope_list = (lines_list[:, 3] - lines_list[:, 1]) / (lines_list[:, 2] - lines_list[:, 0])
        slope_list = np.around(slope_list, 3)

        # Get straight lines with the same slope
        slope_set = set(slope_list)
        slope_dict = {}
        for slope in slope_set:
            temp = [i for i, x in enumerate(slope_list) if slope == x]
            if len(temp) < 2:
                continue
            else:
                slope_dict[slope] = temp
        keys = list(slope_dict.keys())

        for key in range(len(keys)):
            lines = []
            center_point = []

            # Traverse all lines in each group of parallel lines
            for index in slope_dict[keys[key]]:
                x1, y1, x2, y2 = hough_line[index][0]
                lines.append((x1, y1, x2, y2))
                # Save the midpoint of the line
                x_c, y_c = int((x1 + x2) / 2), int((y1 + y2) / 2)
                center_point.append((x_c, y_c))

            for index1, (x1, y1) in enumerate(center_point[:-1]):
                for index2, (x2, y2) in enumerate(center_point[index1 + 1:]):

                    # Calculate the distance between two points
                    point_dis = self.calculate_point_distance(x1, y1, x2, y2)
                    # Calculate the distance from a point to a straight line
                    p1, q1, p2, q2 = lines[index1 + index2 + 1]
                    dropfoot, point_line_dist = self.point2fixedAxis(np.array([x1, y1]), np.array([[p1, q1], [p2, q2]]))

                    if point_dis > 50 or point_dis < 10 or point_line_dist < 10:
                        continue

                    # Determine whether the center line crosses the contour
                    pixels = self.createLineIterator(np.array((x1, y1)), np.array((x2, y2)), mask,
                                                     return_pixels=True)
                    flag = sum(pixels[:, 2])
                    if flag > 0:
                        continue

                    # Determine whether there is a vertical line between the two straight lines(waste time)  t is high，l is width
                    t1, l1, t2, l2 = self.get_dropline(lines[index1], lines[index1 + index2 + 1])

                    # if np.all(img[np.where(cv2.line(img, (t1, l1), (t2, l2), (0, 0, 255), 1))]== np.array(
                    #         [255, 255, 255])):
                    #     continue

                    # 画出轮廓的平行线
                    if t1:
                        # cv2.line(binary_image, (x1, y1), (x2, y2), (b, g, r), 2)
                        # cv2.line(binary_image, (p1, q1), (p2, q2), (b, g, r), 2)
                        #  返回（x1,y1,x2,y2)坐标 patch四个顶点坐标
                        # cv2.line(img, (t1, l1), (t2, l2), (0, 0, 255), 1)
                        res_line.append([t1, l1, t2, l2])
                        res_patch.append([t1 - 32, t2 + 32, l1 - 32, l2 + 32])

                        # cv2.line(img, (t1, l1), (t2, l2), (0, 0, 255), 1)
                        point_distance = self.calculate_point_distance(t1, l1, t2, l2)
                        point_distance_list.append(int(point_distance))
                        print("Find {} lines, the distance between two lines is {}.".format(line_num, point_distance))
                        line_num = line_num + 1
                    # else:
                    # print("Drop line is None")
        return res_line, res_patch

    def calculate_point_distance(self, x1, y1, x2, y2):
        """
        :return: dis
        """
        dis = np.sqrt(np.sum(np.square(abs(x1 - x2)) + np.square(abs(y1 - y2))))
        return dis

    def point2fixedAxis(self, point, fixedAxis):
        """
        Calculate the distance from the point to the straight line, and the dropfoot
        :param point:
        :param fixedAxis:
        :return:
        """
        vector1 = point - fixedAxis[0]
        vector2 = point - fixedAxis[1]
        vector3 = fixedAxis[1] - fixedAxis[0]

        k = np.dot(fixedAxis[0] - point, fixedAxis[1] - fixedAxis[0])
        k /= -np.square(np.linalg.norm(vector3))
        dropFoot = k * vector3 + fixedAxis[0]

        distance = np.linalg.norm(np.cross(vector1, vector2)) / np.linalg.norm(vector3)
        return dropFoot, distance

    def createLineIterator(self, P1, P2, img, return_pixels=False):
        # define local variables for readability
        if return_pixels == True:
            imageH = img.shape[0]
            imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y
        if return_pixels:
            # Remove points outside of image
            colX = itbuffer[:, 0]
            colY = itbuffer[:, 1]
            itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]
            itbuffer = itbuffer[2: -2]

            # Get intensities from img ndarray
            itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]
            return itbuffer
        else:
            return itbuffer

    def get_dropline(self, line1, line2):
        """
        Get the perpendicular line between two parallel lines
        :param line1:
        :param line2:
        :return:
        """
        x1, y1, x2, y2 = line1
        pixels1 = self.createLineIterator(np.array((x1, y1)), np.array((x2, y2)), None, return_pixels=False)
        pixels1 = np.delete(pixels1, -1, axis=1)

        p1, q1, p2, q2 = line2
        pixels2 = self.createLineIterator(np.array((p1, q1)), np.array((p2, q2)), None, return_pixels=False)
        pixels2 = np.delete(pixels2, -1, axis=1)

        for (x1, y1) in pixels1:
            dropfoot, dist = self.point2fixedAxis(np.array([x1, y1]), np.array([[p1, q1], [p2, q2]]))
            mask = np.isin(dropfoot, pixels2)
            if mask.all():
                return int(x1), int(y1), int(dropfoot[0]), int(dropfoot[1])
        return 0, 0, 0, 0
