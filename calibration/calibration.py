# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import math

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
'''
重点参考：

张正友相机标定法（棋盘格标定）
参考：
1、项目仓库地址：https://github.com/Nocami/PythonComputerVision-6-CameraCalibration 
2、相机标定数学原理：
1）重点参考：https://blog.csdn.net/honyniu/article/details/51004397 
2）老师给的项目：https://blog.csdn.net/LuohenYJ/article/details/104697062
3）重点参考：https://zhaoxuhui.top/blog/2018/04/17/CameraCalibration.html
3、亚像素精度定义：https://www.cnblogs.com/Jessica-jie/p/8529564.html#:~:text=%E4%BA%9A%E5%83%8F%E7%B4%A0%E7%B2%BE%E5%BA%A6%E6%98%AF%E6%8C%87,%E4%B9%8B%E9%97%B4%E7%BB%86%E5%88%86%E6%83%85%E5%86%B5%E3%80%82&text=%E8%BF%99%E6%84%8F%E5%91%B3%E7%9D%80%E6%AF%8F%E4%B8%AA,%E8%AF%A5%E7%82%B9%E9%98%B5%E8%BF%9B%E8%A1%8C%E6%8F%92%E5%80%BC%E3%80%82 
4、齐次坐标系：
1）重点参考：https://www.zhihu.com/question/59595799
2）https://www.cnblogs.com/xin-lover/p/9486341.html
5、opencv文档：http://www.opencv.org.cn/opencvdoc/2.3.2/html/index.html

相机标定含义：
相机标定是通过估计相机参数，来对相机镜头进行精度的校准。
每个镜头的畸变程度各不相同，通过相机标定可以校正这种镜头畸变。
其实可以认为用这种标定的方式来求解相机内参和畸变参数，相当于一种相机校准，然后这些参数就可以用于后面的求解。

对于相机参数中外部参数和内部参数的解释：
1、内部参数：相机/镜头系统本身的参数。例如透镜的焦距、光学中心和径向畸变系数。
2、外部参数：这是指相机相对于某些世界坐标系的方位(旋转和平移)。

标定的过程分为两个部分：
1、第一步是从世界坐标系转换为相机坐标系，这一步是三维点到二维点的转换，包括 R RR，t tt （相机外参）等参数
2、第二步是从相机坐标系转为图像坐标系，这一步是二维点到三维点的转换，包括 K KK（相机内参）等参数

亚像素算法的建立和选择：
1、亚像素定位算法的前提条件是：目标不是孤立的单个像素点，而必须是由一定灰度分布和形状分布的一系列像素点组成。
2、一般要经过三个步骤：
1）粗定位过程：
对检测的目标进行初步定位，得到像素的精度；
2）确定搜索范围：
为了提高精度和降低运算量，在目标位置附近选择合适大小的小领域区域作为分析的搜索区域。
3）细定位过程
根据区域特点，选择合适的亚像素算法进行细定位分析以得到亚像素的精度。

代码具体思路：
在实际操作中我们可以使用OpenCV加棋盘格的方法对相机进行标定。一般情况下，用10-20张影像进行标定。

为了找到棋盘的图案，我们使用函数 cv2.findChessboardCorners()。在找到这些角点之后使用函数cv2.cornerSubPix()增加准确度。
然后可以使用函数cv2.drawChessboardCorners()绘制图案。最后得到这些对象点和图像点之后使用cv2.calibrateCamera()进行标定。
在标定结束后，相机的内参矩阵和畸变参数等相关参数会被返回。拿到相机内参、畸变参数等数据后，就可以用它们对相片进行校正，OpenCV中有cv2.undistort()可以很方便地实现这个需求。

在上面的流程中，其实标定的核心函数只是cv2.calibrateCamera()，前面那么多其实都是在做数据的准备。
所以我们完全可以自己编写代码实现或者手工标注进行数据准备，例如有时OpenCV可找不到我们想要的棋盘格的时候。 
OpenCV中校正函数的传入只是对应坐标，所以我们完全可以人工手动标记对应点坐标，再传入函数，同样可以进行标定。只是可能会比较麻烦点。
"""

"""
todo:相机定位重点解决问题：提高角点的定位精度

思路：
1、改代码（减少棋盘格成本）：
1）文件后缀保持原样（.png后缀），裁剪后图片（周围涂白）的相对位置和大小不要更改
2）数格子不方便，可以在代码添加边缘检测+自动裁剪（不要模糊的边界）的流程
3）为了尽量减少畸变，将棋盘格用玻璃片夹住（模拟后续的烟丝检测效果）
2、改棋盘格：定制棋盘格大小/剪裁过大的棋盘格

20210428版本更新：
1.对于棋盘图片的处理，只留下了函数crop_board，最后返回剪裁后并带白边的棋盘格，才能满足粗定位函数的要求
2.因为拍照所用的棋盘是剪裁过后，行和列都是28，在本代码中的凡是棋盘格的维度都写上28，例如cv2.findChessboardCorners()等函数
3.在代码的主函数末尾增加一个for循环，把每个图片的对应的rvecs和tvecs都打印出来方便后续的计算；
4.之前对于旋转角度比较大的棋盘格无法识别角点的处理：
（1）因为之前一直想着通过原先的代码进行图像处理从而数出棋盘维度，但一直数不出来，其实可以直接规定棋盘维度就是28*28
（2）在函数crop_board()内，通过调整识别横线，识别竖线的参数分别为（1，35）和（35,1）可以顺利剪裁棋盘格，即便是旋转角度偏大的
棋盘也可以被剪裁，并且在主函数中可以被粗定位证明剪裁效果是成功的

'''

def crop_board(img):
    """
    根据亮度自动剪裁图片（表格检测原理）
    :输入形式参数 img:输入图像
    :返回return:masked_white_edge 裁剪后带白边的图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img转换为灰度图gray，
    # cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式
    # 参数cv2.COLOR_BGR2GRAY，表示从BGR文件转为灰度图

    # todo:因为亮盘偏转角度较小，初步考虑对点阵描绘最小内接矩形进行剪裁，但后续若要求偏转角度较大，则采用其他方法优化裁剪区域
    #  参考：
    #  1、https://www.cnblogs.com/frombeijingwithlove/p/4226489.html；
    #  2、http://vitrum.github.io/2015/07/28/Opencv-%E5%BA%94%E7%94%A8%EF%BC%8C%E5%9B%BE%E7%89%87%E9%87%8C%E5%8F%96%E5%87%BA%E5%9B%9B%E8%BE%B9%E5%BD%A2/

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    thresh = cv2.adaptiveThreshold(gray, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY, blockSize=5, C=-10)
    # 采用自适应算法对灰度图二值化
    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None)
    # 参数的含义如下：
    # src：灰度化的图片,为上面的变量gray
    # maxValue：满足条件的像素点需要设置的灰度值，255
    # adaptiveMethod：自适应方法。有2种：ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C
    # thresholdType：二值化方法，可以设置为THRESH_BINARY或者THRESH_BINARY_INV
    # blockSize：分割计算的区域大小，取奇数
    # C：常数，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
    # dst：输出图像，可选
    # 输出的thresh是个二值图
    # cv2.imwrite("image/binary.png",thresh)
    # 输出>>>二值化的棋盘格

    # 连通交点
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 函数cv2.getStructuringElement(参数1，参数2)，该函数的返回值是卷积核
    # 第一个参数表示核的形状。可以选择三种
    # 矩形：MORPH_RECT;
    # 交叉形：MORPH_CROSS;
    # 椭圆形：MORPH_ELLIPSE;
    # 第二个参数表示核的尺寸。上面的代码是5x5的矩阵
    # 所以才函数返回的是一个5x5的矩形卷积核

    dilate = cv2.dilate(thresh, dilate_kernel, iterations=2)  # 膨胀
    # cv2.dilate(img, kernel, iteration)
    # img – 目标图片是变量thresh；
    # kernel – 进行操作的内核，是由cv2.getStructringElement函数生成的，默认为3×3的矩阵；但此处的kernel为
    #   上面cv2.getStructuringElement()函数的返回值dilate_kernel
    # iterations – 腐蚀次数，默认为1，但上面为2
    # cv2.imwrite("image/dilate.png", dilate)
    # 输出膨胀过后的棋盘格

    denoise = cv2.medianBlur(dilate, 9)
    # 去除噪声，采用中值滤波
    # 参数1是目标图片
    # 参数2是滤波模板尺寸的大小，必须是大于1的奇数
    # 输出denoise变量为被滤波过后的图片
    # cv2.imwrite("image/denoise.png", denoise)

    # 识别横线 >>>>> 原先参数（200,1）-> (35,1)
    row_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    # 返回一个35x1的矩形卷积核row_kernel,也就是一条横线
    eroded_row = cv2.erode(denoise, row_kernel, iterations=1)  # 侵蚀
    # cv2.imwrite("image/row.png", eroded_row) 这条注释掉的代码是用来观察侵蚀后的图片的

    # 识别竖线>>>>>>原先参数（1,200） -> (1,35)
    col_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    # 返回一个1x35的矩形卷积核col_kernel也就是一条竖线
    eroded_col = cv2.erode(denoise, col_kernel, iterations=1)  # 侵蚀
    # cv2.imwrite("image/col.png", eroded_col) 理由同line 80

    # 标识交点
    # todo:交叉点太小了，可以考虑继续用膨胀
    bitwise_and = cv2.bitwise_and(eroded_row, eroded_col)
    # 位与运算
    # cv2.bitwise_and()是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作
    # cv2.imwrite("image/bitwise_and.png", bitwise_and)

    # 在原图上显示标记点
    # tag_point = cv2.add(gray, bitwise_and)  # 合并两个图片
    # cv2.imwrite("image/tag_point.png", tag_point)

    # 识别黑白图中的白色交叉点，将横纵坐标取出
    ys, xs = np.where(bitwise_and > 0)
    #np.where的解释：
    # 1.np.where(condition,x,y)
    # 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
    # 2.np.where(condition)
    # 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式

    points = []  # 收集识别出的交叉点坐标（可能由多个元素组成）
    for i in range(len(xs)):
        points.append((xs[i], ys[i]))

    hull_cnt = cv2.convexHull(np.array(points))
    # 先找凸包，给定二维平面上的点集，凸包就是将最外层的点连接起来构成的凸多边形，它能包含点集中所有的点。凸包即为凸多边形
    # hull = cv2.convexHull(points, clockwise, returnpoints)的参数说明：
    # hull : 输出凸包结果，n * 1 *2 数据结构，n为外包围圈点数
    # points: 输入的坐标点，通常为1* n * 2 结构，n为所有的坐标点的数目
    # clockwise：转动方向，TRUE为顺时针，否则为逆时针；
    # returnPoints：默认为TRUE，返回凸包上点的坐标，如果设置为FALSE，会返回与凸包点对应的轮廓上的点

    # 获取最小外接矩阵，中心点坐标，宽高，旋转角度
    # 参考：
    # 1、重点：https://blog.csdn.net/lanyuelvyun/article/details/76614872#:~:text=5374-,%E4%BD%BF%E7%94%A8python%20opencv%E8%BF%94%E5%9B%9E%E7%82%B9%E9%9B%86cnt%E7%9A%84%E6%9C%80%E5%B0%8F%E5%A4%96%E6%8E%A5,%E7%82%B9%E9%9B%86%E4%B8%8D%E5%AE%9A%E4%B8%AA%E6%95%B0%E3%80%82&text=%E8%AF%A5%E5%87%BD%E6%95%B0%E8%AE%A1%E7%AE%97%E5%B9%B6%E8%BF%94%E5%9B%9E,%E6%9C%80%E5%B0%8F%E5%8C%BA%E5%9F%9F%E8%BE%B9%E7%95%8C%E6%96%9C%E7%9F%A9%E5%BD%A2%E3%80%82
    # 2、重点：https://blog.csdn.net/Maisie_Nan/article/details/105833892
    # 3、围绕中心缩放矩形（思路：先按左上角点缩放，再整体平移）：https://blog.csdn.net/kh1445291129/article/details/51149849

    cut_distance = -100# 控制矩形缩放距离

    (center_x, center_y), (box_width, box_height), theta = cv2.minAreaRect(hull_cnt)
    # 获取最小外接矩阵，返回值：(中心点坐标x,中心点坐标y)，(宽,高)，旋转角度
    rectangle = cv2.boxPoints(
        ((center_x, center_y), (box_width - cut_distance, box_height - cut_distance), theta))
    # 围绕中心点缩放矩形，长/宽缩放distance长度
    # 获取最小外接矩形的四个顶点
    rectangle = np.int0(rectangle)  # 取整（因为像素值为整数）

    imcopy = img.copy()  # 防止对原图破坏
    cv2.drawContours(imcopy, [rectangle], 0, (0, 0, 255), 15)
    # 使用轮廓绘制函数cv2.drawContours()，画出圈定裁剪区域
    # cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
    # 第一个参数image,是指明在哪幅图像上绘制轮廓；image为三通道(RGB图)才能显示轮廓
    # 第二个参数是轮廓本身，在Python中是一个list列表;
    # 第三个参数指定绘制轮廓list中的哪条轮廓，如果是 - 1，则绘制其中的所有轮廓。
    # 后面的参数很简单。其中thickness表明轮廓线的宽度，如果是 - 1（cv2.FILLED），则为填充模式。

    # cv2.imwrite('image/rectangle.png', imcopy)

    # 裁剪图像，加上白边
    # 参考：
    # 1、图像位运算应用蒙版：https://www.geeksforgeeks.org/arithmetic-operations-on-images-using-opencv-set-2-bitwise-operations-on-binary-images/
    # 2、对彩色图像应用蒙版：https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image
    # 3、对图像应用蒙版时，numpy的数据类型问题：https://stackoverflow.com/questions/49970147/opencv-masking-image-error-215-mtype-0-mtype-1-mask-samesiz
    # 4、numpy数据类型（选择uint8类型，是因为灰度图的灰度范围为0-255）：https://numpy.org/doc/stable/user/basics.types.html
    crop_mask = np.zeros(gray.shape, dtype=np.uint8)  # 设置蒙版（注意numpy数据类型）
    # 返回来一个给定形状和类型的用0填充的数组；
    cv2.fillConvexPoly(crop_mask, rectangle, 255)  # 填充多边形蒙版
    # cv2.fillConvexPoly( image , 多边形顶点array , RGB color)函数可以用来填充凸多边形,只需要提供凸多边形的顶点即可.
    # 其中RGB color在函数中为255，即为白色，也就是用白色来填充
    masked = cv2.bitwise_and(img, img, mask=crop_mask)
    # 裁剪棋盘格，这句代码的解释参考：https://stackoverflow.com/questions/32774956/explain-arguments-meaning-in-res-cv2-bitwise-andimg-img-mask-mask
    crop_mask_inv = cv2.cvtColor(~crop_mask, cv2.COLOR_GRAY2BGR)  # 蒙版从灰度图变为BGR色彩空间
    masked_white_edge = cv2.bitwise_xor(masked, crop_mask_inv)  # 在裁剪的棋盘格周围涂上白色背景

    # cv2.imwrite("image/crop.png", masked_white_edge)

    return masked_white_edge

# def getDist_P2P(image,eval):
#     distance = math.pow((np.squeeze(image) - np.squeeze(eval)), 2)
#     #上面math.pow实现了point0[0]-pointA[0]的差的平方
#     distance = math.sqrt(distance)
#     #对distance开方
#     return distance

# ------------函数：计算欧氏距离----------
def distEclud(vecA, vecB):
    dis_list = []
    for i in range(len(vecA)):
        dis = np.sqrt(np.sum((np.power((vecA[i] - vecB[i]), 2))))
        dis_list.append(dis)
    # return np.sqrt((np.power((vecA - vecB), 2)))
    return dis_list


#                                            #
# -----------------主函数---------------------#
#                                            #
# 设置寻找亚像素角点的参数，采用的运行准则（criteria）是最大循环次数30和最大误差容限0.001，即用来控制算法的迭代数和精确度
# 参考：https://www.jianshu.com/p/b24086aab3fc
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 获取标定板角点的世界坐标
# 注意棋盘格一格5毫米的精度，world_points为np.float32类型，要提高精度
gird_length = 5  # 棋盘格一个格子的边长（5mm）
col_count = 28 #列的维度
row_count = 28 #行的维度
coordinate_error = 0    #根据热图显示得到的坐标误差
camera_points = []  # 存储棋盘格图像的3D点矢量，相机/现实世界的坐标系
image_points = []  # 存储棋盘格图像的2D点矢量，图像坐标系（10-20张图的细定位角点，用于不断迭代提高精确度）
#                                                          #
# <<<<<<<<<<<<<<<<下面为多张棋盘格循环的代码：>>>>>>>>>>>>>>>>>#
#                                                          #
img_paths = glob.glob('4_30/*.png')  # 批量处理图片，获得更精确的相机内参数
# img_paths = glob.glob('E:\pycharm\company-camera-calibration\image\\*.png')
for img_path in img_paths:
    img = cv2.imread(img_path)  # 获取image文件夹中的棋盘格图像

    # 图像预处理
    crop_img = crop_board(img)  # 根据亮度自动剪裁图片（表格检测原理），调用了上面定义的crop_board(img)函数
    # thres_img = thres_board(crop_img)  # 剪裁后的棋盘二值化，便于识别角点，不受剪裁大小的影响，调用了上面定义的thres_board(img)函数
    # row_count, col_count = count_dimension(crop_img)  # 自动找寻点阵的纵向与横向点的个数，调用上面定义的count_dimension函数

    world_points = np.zeros((col_count * row_count, 3), np.float32)  # 初始化棋盘格上角点的世界坐标（三维）

    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    # np.mgrid[start:end:step]
    # start:开始坐标
    # stop:结束坐标（实数不包括，复数包括
    # step:步长
    world_points[:, :2] = np.mgrid[coordinate_error * gird_length:(row_count + coordinate_error) * gird_length:gird_length,
                          coordinate_error * gird_length:(col_count + coordinate_error) * gird_length:gird_length].T.reshape(-1, 2)
    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y（注意间隔5mm）
    # world_points[:, :2] = np.mgrid[-14 * 5:28 * gird_length - 14 * 5:gird_length,
    #                       -14 * 5:28 * gird_length - 14 * 5:gird_length].T.reshape(
    #     -1,
    #     2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y（注意间隔5mm）

    # 寻找棋盘角点的坐标，(col_count, row_count)是棋盘格的维度，该方法要求棋盘格周围有白边，以便棋盘格能从背景中分割出来
    # cv2.findChessboardCorners()只是粗定位角点的坐标，需要配合cv2.cornerSubPix进行细定位分析，获得精度更高的角点坐标（亚像素级别）
    # cv2.findChessboardCorners()根据图像的旋转方向，起始角点可能从左上角或者右下角开始标注，然后标注的顺序是先按行从上往下（或从下往上），再按列从左往右（或从右往左），参考：https://stackoverflow.com/questions/19190484/what-is-the-opencv-findchessboardcorners-convention
    # 如果不能找到所有的角点，则返回值patternfound为0，表示没有找到，否则为1；coarse_corners即粗定位角点的相机坐标
    patternfound, coarse_corners = cv2.findChessboardCorners(crop_img, (row_count, col_count),
                                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # 如果粗定位角点坐标成功，进一步对角点细定位，通过粗定位来筛选棋盘格和其他非棋盘格的图片，只有棋盘格能被粗定位到角点，再进一步细定位

    if patternfound:
        camera_points.append(world_points)

        # print("camera points:\n",camera_points)

        # 函数不断迭代来细化角点的定位（更精确），求解亚像素精度
        # todo:函数所用的数学原理貌似是多项式插值法？参数解释？
        # image 输入图像（建议灰度图，不要用二值化的图）；corners 粗定位角点坐标；winSize 搜索窗边长的一半（搜索窗用于在粗定位角点基础上，不断迭代找寻粗角点的邻域中更为精确的角点位置，搜索窗的大小人为控制，不易过大或过小）
        # zeroZone todo:啥意思?
        # criteria 算法迭代准则
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        detail_corners = cv2.cornerSubPix(image=gray, corners=coarse_corners, winSize=(5, 5), zeroZone=(-1, -1),
                                          criteria=criteria)  # 在原角点（粗定位）的基础上寻找亚像素角点

        # image_points存储10-20张细定位的角点（array of list），用于不断迭代提高精度
        # 如果细定位的亚像素点找到，则以此归入图像坐标系，否则用粗定位的像素点归入图像坐标系
        # np.array比较大小，需要用特殊的判断，即np.any()或np.all()，参考：https://blog.csdn.net/sinat_33563325/article/details/79868109
        image_points.append(detail_corners)

        # # 在原图上绘制定位的角点
        # cv2.drawChessboardCorners(img, (row_count, col_count), coarse_corners,
        #                           patternfound)  # 绘制棋盘格上粗定位的角点，记住，OpenCV的绘制函数一般无返回值
        # cv2.imwrite('image/calibration.png', img)  # 存储图片

# 获取内参、外参
# patternfound todo:粗定位角点的置信度？
# camera_matrix 内参数矩阵；dist_coeff 畸变系数（径向畸变、切向畸变的5个参数）；rvecs 旋转向量（外参数）；tvecs 平移向量（外参数）
# todo:cv2.calibrateCamera()的具体算法步骤？参考：http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=calibratecamera#calibratecamera
img = cv2.imread(img_paths[0])  # 获取image文件夹中的棋盘格图像
img_size = img.shape[:2]  # 获取图像大小（单位：像素），python索引切片参考：https://www.jianshu.com/p/d3839175eaf4，todo:长*宽变为宽*长？
patternfound, camera_matrix, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(camera_points, image_points,
                                            img_size,
                                            None,
                                            None)  # 通过已定位的角点（相机定位），找寻定位的外参数和内参数（获得校准参数）
print("相机内参数矩阵camera_matrix：\n", camera_matrix)
print("畸变系数dist_coeff(内参)：\n", dist_coeff)
# print("旋转向量（外参数）rvecs：\n", rvecs)
# print("平移向量（外参数）tvecs：\n", tvecs)

print("<<<<<<<<<<<<<<<<<<<<<以下为各个棋盘格图片对应的参数>>>>>>>>>>>>>>>>>>>>>>>")

#-------新增加的循环:用于输出每张棋盘格的各个参数------
total_error = 0  #整体误差
dis = 0          #image_points 与 eval_points的距离
eval_points_list = []
dis_lists = []     #用于存放eval和image的误差距离的数组

#以下的循环注意image_points要有[i]，而输出的eval_points是由image_points算出来的
#但在循环中eval_points不需要加[i]在后面
for i in range(len(image_points)):
    #以下输出图片i的image_points和image_points的坐标个数：
    length_image = len(image_points[i])
    print("image_point"+str([i])+"有："+str(length_image)+"个坐标")
    print("image_point" + str([i]) + ":\n", image_points[i])


    print("旋转向量(内参数)rvecs"+str(i)+":\n", rvecs[i])
    print("平移向量（外参数）tvecs"+str(i)+":\n", tvecs[i])
    #计算eval_points:
    eval_points, _ = cv2.projectPoints(camera_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeff)
    # eval_points_list.append(eval_points)
    # eval_points_list[i] 引用eval_points的方法

    #以下输出图片i的eval_points和坐标个数
    length_eval = len(eval_points)
    print("eval point" + str([i]) + "有：" + str(length_eval) + "个坐标")
    print("eval_point"+str([i])+":\n"+str(eval_points)) #显示eval_points的坐标
    # print("eval_point" + str([i]) + ":\n" + str(np.squeeze(eval_points)))

    # 计算反投影误差：通过之前计算的内参数矩阵、畸变系数、旋转矩阵和平移向量，使用cv2.projectPoints()计算三维点到二维图像的投影，
    # 然后计算反投影得到的点与图像上检测到的点的误差，最后计算一个对于所有标定图像的平均误差即反投影误差
    error = cv2.norm(image_points[i], eval_points, cv2.NORM_L2) / len(image_points)
    total_error += error

    #计算每个图片中每个image_point与对应的eval_point的误差dis，dis是个数组
    #每张图片的image_points有748个,正好是28*28个点，所以可以排列成28*28的矩阵，reshape（28,28）
    dis_lists = distEclud(np.squeeze(image_points[i]), np.squeeze(eval_points))

    # print("有"+str(len(dis_lists))+"个坐标差")
    dis_lists = np.array(dis_lists)
    # print("误差数组：\n", dis_lists)
    err_map = dis_lists.reshape(28, 28)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(err_map, cmap=plt.cm.hot_r)
    plt.show()


    # print("image" + str(i) + "的平均误差距离：", dis/length)

    print("<<<<<<<<<<<<<<<<这是个间隔,方便观察>>>>>>>>>>>>>>>>>>>>")
#总结输出总误差
print("平均误差total error: ", total_error/len(image_points))








