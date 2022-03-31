import configs
import cv2


class Detector(object):
    def __init__(self):
        r"""parameter initialization
        
        """
        self.cfg = configs.get_config()

    def preprocess(self, img):
        r"""shared preprocessing of image between detectors
        
        Args:
            img: cv2 or numpy data with shape [3036, 4024, 3] and intensity as uint8 in [0, 255]
            
        Return:
            img: preprocessed image.
            
            binary: binary image.
            
            contours: searched contours by opencv.
        """
        # cut out dark border, replace it with pure white,the white size[265,265,770,770]
        img = img[self.cfg.detector.cut_vet:-self.cfg.detector.cut_bot,
              self.cfg.detector.cut_hor:-self.cfg.detector.cut_hor, :]
        # img = cv2.copyMakeBorder(img, self.cfg.detector.pad, self.cfg.detector.pad, self.cfg.detector.pad, self.cfg.detector.pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img = cv2.copyMakeBorder(img, self.cfg.detector.cut_vet, self.cfg.detector.cut_bot, self.cfg.detector.cut_hor,
                                 self.cfg.detector.cut_hor, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # convert to binary image
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours_res = []
        for c in range(len(contours)):
            # 计算每一个轮廓的面积
            # area = cv2.contourArea(contours[c])
            # 计算每一个轮廓的弧长
            arclen = cv2.arcLength(contours[c], True)
            # 设置一定的阈值过滤部分轮廓
            if arclen > self.cfg.detector.cont_len_thres:
                contours_res.append(contours[c])

        return img, binary, contours_res

    def run(self,  binary, contours):
        r"""process a image, return detection results
        
        Args:
            binary: cv2 or numpy data with shape [3036, 4024] and intensity as uint8 in [0, 255]

            contours: contours found by cv2
            
        Return:
            lines: detected lines for width measuring.list shape [n],[4] where n is number of lines. Each line with two points as [x0, y0, x1, y1]. Coordinates are at the original image before preprocessing!!
            
            patches: detected patches for each line.list shape [n],[4]. Each patch with coordinates as [top_left_x, top_left_y, bottom_right_x, bottom_right_y]. Coordinates are at the original image before preprocessing!!
        """
        raise NotImplementedError
