from os.path import exists
from urllib.request import urlretrieve
import time
import cv2
import numpy as np

prototxt_thermal = r"/home/bestlab/Desktop/final_com/model/thermal.prototxt"
caffemodel_thermal = r"/home/bestlab/Desktop/final_com/model/thermal.caffemodel"

net_thermal = cv2.dnn.readNetFromCaffe(
    prototxt = prototxt_thermal, caffeModel = caffemodel_thermal)

def detect_thermal(img, min_confidence=0.5):
    # 取得img的大小(高，寬)

    (h, w) = img.shape[:2]
    inHeight = 300
    inWidth = 300
    inScaleFactor = 2/255
    meanVal = 127.5
    # 建立模型使用的Input資料blob (比例變更為300 x 300)
    blob = cv2.dnn.blobFromImage(
        img, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal))

    # 設定Input資料與取得模型預測結果
    net_thermal.setInput(blob)
    detectors = net_thermal.forward()
    # 初始化結果
    rects = []

    # loop所有預測結果
    for i in range(0, detectors.shape[2]):
        # 取得預測準確度
        confidence = detectors[0, 0, i, 2]

        # 篩選準確度低於argument設定的值
        if confidence < min_confidence:
            continue

        # 計算bounding box(邊界框)與準確率 - 取得(左上X，左上Y，右下X，右下Y)的值 (記得轉換回原始image的大小)
        box = detectors[0, 0, i, 3:7] * np.array([w, h, w, h])
        # 將邊界框轉成正整數，方便畫圖
        (x0, y0, x1, y1) = box.astype("int")
        rects.append((x0, y0, x1 - x0, y1 - y0,confidence))
    return rects
