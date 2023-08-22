import cv2 as cv
import numpy as np


def dilate_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.dilate(binary, kernel)
    cv.imshow("dilate_demo", dst)

# original image
# src = cv.imread("I://dataset//EUS_bulk//train//P//EUS1_p0_img_1.jpg")

# marker image
src = cv.imread('cropped.jpg')


cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)    #创建GUI窗口,形式为自适应
cv.imshow("input image", src)    #通过名字将图像和窗口联系

dilate_demo(src)

cv.waitKey(0)   #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
cv.destroyAllWindows()  #销毁所有窗口
