#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import the necessary packages


import numpy as np
import dlib
import cv2
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist


# 论文 Real-Time Eye Blink Detection using Facial Landmarks
def eye_aspect_ratio(eye):
    # 眼睛垂直距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 眼睛水平距离
    C = dist.euclidean(eye[0], eye[3])
    # 使用两者的比值来度量眨眼
    ear = (A + B) / (2.0 * C)
    return ear


# 眼睛长宽比阈值
EYE_AR_THRESH = 0.2
# 噪声阈值 -- 可自行调节
# 自测发现 2 帧时检测眨眼效果比 3 帧好，但噪声较大
EYE_AR_CONSEC_FRAMES = 2
# 疲劳阈值
EYE_AR_CONSEC_FRAMES_SLEEP = 10

# 统计噪声
COUNTER = 0
# 眨眼次数
TOTAL = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 左右眼索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # 转换成灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        # 找到 bindingbox
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # 双眼平均
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


        # 疲劳检测
        if COUNTER >= EYE_AR_CONSEC_FRAMES_SLEEP:
            cv2.putText(frame, "DANGER!!!", (70,320), 0, 3, (0,0,255), 5)


        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # 消除抖动噪声
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            COUNTER = 0


    cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
