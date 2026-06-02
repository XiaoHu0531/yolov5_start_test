"""
    对检测到的“fire”目标物进行圆形相似度判断
    image： “fire”目标框
    sim： 输出的圆形相似度
"""
import time

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

def check_fire(srcimg,det):
    if det[5] != 1:
        return int(det[2] -det[0]) * int((det[3]-det[1]))
    obj_det = [int(element) for element in det[:4]]  # 转为in
    x1, x2, y1, y2 = obj_det[1], obj_det[3], obj_det[0], obj_det[2]
    obj_img = srcimg[x1:x2, y1:y2]
    #讲输入的图像转换为灰度图
    gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 5), 0)
    #对比灰度图使用Ostu算法
    ret1, th1 = cv2.threshold(blur, 0,255,cv2.THRESH_OTSU)
    #核的大小
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    #轮廓提取
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours):
        max_contour = max(contours, key=cv2.contourArea)  #按面积提取最大轮廓
        epsilon = 0.0015 * cv2.arcLength(max_contour,True)
        approx = cv2.approxPolyDP(max_contour,epsilon,True)
        area = cv2.contourArea(approx)
        return area

    else:
        return int(obj_det[2] -obj_det[0]) * int((obj_det[3]-obj_det[1])) # w * h

if __name__ == '__main__':

    image = cv2.imread(r"/media/ps/Linux_Gutail2/hx/fire_dataset/fire/images (2).jpg")
    t1 = time.perf_counter()
    sim = check_fire(image)
    t2 = time.perf_counter()
    print("sim:==>",sim)
    print("usetime:==>",t2-t1)
