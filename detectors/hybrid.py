import detectors
import numpy as np
import cv2
import utils.tool as util
import time

class Hybrid(detectors.Detector):
    
    def __init__(self):
        r"""parameter initialization
        
        """
        super(Hybrid, self).__init__()
        
    def dist_point_line(self, point, line, slope):
        re_line = [point[0], point[1]]
        
        if slope == 0.0:
            dist = abs(line[1]-point[1])
            re_line.append(point[0])
            re_line.append(line[1])
            
            return dist, re_line
            
        if slope == float("inf") or slope == float("-inf"):
            dist = abs(line[0]-point[0])
            re_line.append(line[0])
            re_line.append(point[1])
            
            return dist, re_line
        
        slope_v = -1.0/slope
        b1 = point[1]-slope_v*point[0]
        b2 = line[1]-slope*line[0]        
        x = -(b1-b2)/(slope_v-slope)
        y = slope_v*x+b1
        
        dist = np.sqrt(np.square(point[0]-x)+np.square(point[1]-y))
        re_line.append(x)
        re_line.append(y)
        
        return dist, [re_line[0], re_line[1], round(re_line[2], 0).astype(np.int32), round(re_line[3], 0).astype(np.int32)]
    
    def get_near_point(self, line1, line2):
        delta11 = abs(line1[0]-line2[0])+abs(line1[1]-line2[1])
        delta12 = abs(line1[0]-line2[2])+abs(line1[1]-line2[3])
        delta21 = abs(line1[2]-line2[0])+abs(line1[3]-line2[1])
        delta22 = abs(line1[2]-line2[2])+abs(line1[3]-line2[3])
        
        dists = [delta11, delta12, delta21, delta22]
        
        idx = dists.index(min(dists))
        
        if idx == 0:
            return [line1[0], line1[1]], [line2[0], line2[1]]
        
        if idx == 1:
            return [line1[0], line1[1]], [line2[2], line2[3]]
        
        if idx == 2:
            return [line1[2], line1[3]], [line2[0], line2[1]]

        if idx == 3:
            return [line1[2], line1[3]], [line2[2], line2[3]]
        
        return            
        
    def dist_parallel(self, line1, line2, slope):
        mid_p_line1 = [(line1[0]+line1[2])/2.0, (line1[1]+line1[3])/2.0]
        mid_p_line2 = [(line2[0]+line2[2])/2.0, (line2[1]+line2[3])/2.0]
        
        point1, point2 = self.get_near_point(line1, line2)        
            
        dist, re_line = self.dist_point_line(point1, line2, slope)
        
        if re_line[2] < min(line2[0], line2[2]) or re_line[2] > max(line2[0], line2[2]):
            dist = 0
            
        if re_line[3] < min(line2[1], line2[3]) or re_line[3] > max(line2[1], line2[3]):
            dist = 0
            
        if dist <= 0:
            dist, re_line = self.dist_point_line(point2, line1, slope)

            if re_line[2] < min(line1[0], line1[2]) or re_line[2] > max(line1[0], line1[2]):
                dist = 0
            
            if re_line[3] < min(line1[1], line1[3]) or re_line[3] > max(line1[1], line1[3]):
                dist = 0
                            
        return dist, re_line
    
    def find_width_out(self, p, mask, method):
        if method == 'once':
            re, patch = util.find_width(p, mask, padding=self.cfg.hybrid.patch_pad)
            return re, patch, util.dist(re)
    
        if method == 'twice':
            re1, patch1 = util.find_width(p, mask, padding=self.cfg.hybrid.patch_pad)
            if re1 is None:
                return re1, patch1, None
    
            p2 = [re1[2], re1[3]]
            re2, patch2 = util.find_width(p2, mask, padding=self.cfg.hybrid.patch_pad)
    
            if re2 is None:
                return re1, patch1, util.dist(re1)
    
            if util.dist_sq(re2) > util.dist_sq(re1):
                return re1, patch1, util.dist(re1)
            else:
                return re2, patch2, util.dist(re2)
    
        if method == 'iter':
            re, patch = util.find_width(p, mask, padding=self.cfg.hybrid.patch_pad)
            if re is None:
                return re, patch, None
    
            # i = 1
            while True:
                re_n, patch_n = util.find_width([re[2], re[3]], mask, padding=self.cfg.hybrid.patch_pad)
    
                if re_n is None or util.dist_sq(re_n) >= util.dist_sq(re):
                    return re, patch, util.dist(re)
    
                re = re_n
                patch = patch_n
                # print(i)
                # i = i+1
    
        return None, None, None
        
    def run(self, binary, contours):
        re_lines = []
        re_patches = []
        mask_shared = np.zeros(binary.shape, dtype=np.uint8)+255
        num_shared = 0
        
        # parallel algorithm
        start = time.time()
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 1, cv2.LINE_8)
        binary = 255-binary
        binary[np.where(mask == 255)] = 255
        
        mask_slopes_ids = np.zeros((binary.shape[0], binary.shape[1], 2), dtype=np.float32)-5000.0
        # test = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)    
        
        hough_line = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180,
                                     threshold=5, minLineLength=3, maxLineGap=2)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        lines_list = hough_line.reshape(-1, 4)
        slope_list = (lines_list[:, 3] - lines_list[:, 1]) / (lines_list[:, 2] - lines_list[:, 0])
        slope_list = np.around(slope_list, 3)
        
        # for i in range(lines_list.shape[0]):
        #     cv2.line(test, (lines_list[i, 0], lines_list[i, 1]), (lines_list[i, 2], lines_list[i, 3]), (0, 0, 255), 1)
            
        # cv2.imwrite('tmp.png', test)
        
        for i in range(lines_list.shape[0]):
            line = lines_list[i, :]
            
            cv2.line(mask_slopes_ids, (line[0], line[1]), (line[2], line[3]), (slope_list[i], i), 1)
            # cv2.line(test, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)
            # cv2.line(mask, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)        
        
        for i in range(lines_list.shape[0]):
            line = lines_list[i, :]
            
            if mask_shared[line[1], line[0]] == 0 or mask_shared[line[3], line[2]] == 0:
                continue
            
            patch = util.get_patch_with_line_rad(mask_slopes_ids, line, self.cfg.hybrid.patch_size)
            
            diff_line = np.where((patch[:, :, 0] == slope_list[i]) & (patch[:, :, 1] != i))
            
            if len(diff_line[0]) > 0:
                ret_ids = []
                
                for j in range(len(diff_line[0])):
                    id = patch[diff_line[0][j], diff_line[1][j], 1].astype(np.int32)
                    if ret_ids.count(id) > 0:
                        continue
                    
                    ret_ids.append(id)
                
                    line2 = lines_list[id, :]
                    
                    dist, re_line = self.dist_parallel(line, line2, slope_list[i])
                    
                    if dist < self.cfg.hybrid.dist_thres_low or dist > self.cfg.hybrid.dist_thres_high:
                        continue
                    
                    cv2.line(mask_shared, (re_line[0], re_line[1]), (re_line[2], re_line[3]), (0, 0, 0), 1)
                    # cv2.line(mask, (re_line[0], re_line[1]), (re_line[2], re_line[3]), (0, 255, 0), 1)
                    # cv2.imwrite('tmp.png', mask)
                    
                    left = min(re_line[0], re_line[2])-1
                    right = max(re_line[0], re_line[2])+1
                    top = min(re_line[1], re_line[3])-1
                    bottom = max(re_line[1], re_line[3])+1
                    
                    out_ids = np.where((mask_shared[top:bottom, left: right] == 0) & (binary[top: bottom, left: right] == 0))
                    
                    if len(out_ids[0]) > 0:
                        continue
                    
                    num_shared = num_shared+1
                    print(num_shared)

                    center = (round((left+right)/2), round((top+bottom)/2))
                    left = center[0]-self.cfg.hybrid.patch_pad
                    right = center[0]+self.cfg.hybrid.patch_pad
                    top = center[1]-self.cfg.hybrid.patch_pad
                    bottom = center[1]+self.cfg.hybrid.patch_pad
                    
                    mask_shared[top:bottom, left: right] = 0
                    
                    re_lines.append(re_line)
                    re_patches.append([top, bottom, left, right])   
                
        # print(time.time() - start)

        # skeleton algorithm
        # start = time.time()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
        mask[mask == 255] = 10
        mask[mask == 0] = 255
        mask[(labels != 0) & (mask != 10)] = 128
        
        skeleton = cv2.ximgproc.thinning(binary)
        mask[skeleton == 255] = 0
        
        indexes = list(zip(*np.where(mask == 10)))
        np.random.shuffle(indexes)

        for y, x in indexes:
            if mask_shared[y, x] == 0:
                continue
            
            line, patch, dist = self.find_width_out((x, y), mask, self.cfg.hybrid.method)
            
            if line is None:
                continue
            
            if dist > self.cfg.hybrid.dist_thres_high or dist < self.cfg.hybrid.dist_thres_low:
                continue
            
            num_shared = num_shared+1
            print(num_shared)
            
            top, bottom, left, right = patch
            line = [line[0] + left, line[1] + top, line[2] + left, line[3] + top]
            mask_shared[top : bottom, left : right] = 0

            re_lines.append(line)
            re_patches.append(patch)
            
            if len(re_lines) > self.cfg.hybrid.max_res:
                break

        print('time: %f' % (time.time()-start))

        return re_lines, re_patches
