# -*- coding: utf-8 -*-
# @Time    : 2017/7/27 10:57
# @Author  : play4fun
# @File    : HoughCircles_eyes.py
# @Software: PyCharm

"""
HoughCircles_eyes.py:
http://blog.csdn.net/on2way/article/details/47028969
官方中文文档霍夫圆检测：http://www.woshicver.com/FifthSection/4_14_%E9%9C%8D%E5%A4%AB%E5%9C%88%E5%8F%98%E6%8D%A2/

霍夫梯度法的思路：
1. 第一步根据每个点的模向量来找到圆心，这样三维的累加平面就转化为二维累加平面；
2. 第二步根据所有候选中心的边缘非零像素对其的支持程度来确定半径。

OpenCV内的HoughCircles对基础的Hough变换找圆做了一定的优化来提高速度，它不再是在参数空间画出一个完整的圆来进行投票，
而只是计算轮廓点处的梯度向量，然后根据搜索的半径R在该梯度方向距离轮廓点距离R的两边各投一点，最后根据投票结果图确定圆心位置
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def hough_circle(img, **kwargs):
    # if kwargs['single_image']==True:
    # img_name = kwargs['img_name']
    # index = kwargs['index']
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
    gray = img

    plt.subplot(121), plt.imshow(gray, 'gray')
    plt.xticks([]), plt.yticks([])
    circles = None
    # hough transform   #规定检测的圆的最大最小半径，不能盲目的检测，否侧浪费时间空间。
    # circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,100, param1=100, param2=30, minRadius=200, maxRadius=300)
    circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=0,
                                maxRadius=30)  # 把半径范围调小点，检测内圆,瞳孔
    if circles1 is not None:
        circles = circles1[0, :, :]  # 提取为二维
        circles = np.uint16(np.around(circles))  # 四舍五入，取整
        for i in circles[:]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)  # 画圆
            cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)  # 画圆心
    else: pass
    plt.subplot(122), plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.imsave(kwargs['save_path']+f'/hough_1.jpg', img)
    plt.close()
    # if index==0:
    #     print('其中circles[0]，circles[1]为圆的中心坐标，而circles[2]为圆的半径')
    return circles


if __name__ == '__main__':
    img = cv2.imread('./image.jpg')
    # img = cv2.imread('I://dataset//EUS_bulk//train//P//EUS1_p0_img_1.jpg')
    circles = hough_circle(img)
