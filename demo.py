#!/usr/bin/env python
#-*- coding:UTF-8-*-


import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkFont
from PIL import Image, ImageTk
from skimage import io
import dlib
import numpy as np

#查看opencv的版本
#print(cv2.__version__)

# 是否开启黑色背景
gsetblackimage = False

# 基准人脸的特征
gBaseFaceFeather = np.zeros((128,))

#Set up GUI
window = tk.Tk()  #Makes main window
#window.geometry('890x530')
window.wm_title("人脸识别")
window.config(background = "#FFFFFF")

#设置人脸检测与对齐的参数
detector = dlib.get_frontal_face_detector()
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
faceshape = dlib.shape_predictor("modules/shape_predictor_68_face_landmarks.dat")
# You can download a trained facial shape predictor and recognition model from:\n"
# http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
# http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
facerec = dlib.face_recognition_model_v1("modules/dlib_face_recognition_resnet_model_v1.dat")

# 摄像头标签
#cameraLable = tk.Label(window, text = '监控画面', font = '"Dejavu san" -12 bold', fg = 'green', bg = 'white')
cameraLable = tk.Label(window, text = '监控画面',  fg = 'green', bg = 'white')
# cameraLable.config(width = 200, height=1)
cameraLable.grid(row = 0, column = 0, padx = 10, pady = 10)

#Graphics window
imageFrame = tk.Frame(window, width = 640, height = 480)
# imageFrame.grid(row = 1, column = 0, rowspan = 4, padx = 10, pady = 0)
imageFrame.grid(row = 1, column = 0, rowspan = 4, columnspan = 1, padx = 10, pady = 0)


#检测、对齐、识别人脸
def dectAligRegFace(image, dectlandmark = True, dectfeature = False):

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    #detFaces = detector(image, 1)
    detFaces = detector(image, 0)
    print("Number of faces detected: {}".format(len(detFaces)))
    facebox = []
    landmark = []
    face_descriptor = []
    for i, d in enumerate(detFaces):
        # dlib检测出来的矩形是左上、右下两个个角点的坐标
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    i, d.left(), d.top(), d.right(), d.bottom()))
        #cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 255), 1)
        #cv2.putText(image, 'fece {}'.format(i + 1), (d.left(), d.top() - 4), 0, 0.7, (0,0,255),1)
        facebox += [d.left(), d.top(), d.right(), d.bottom()]

        # 人脸特征点的提取
        if dectlandmark or dectfeature:
            shape = faceshape(image, d)
            #shape.part(0)是一个元组包含特征点的横坐标和纵坐标
            #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
            for point in range(68):
                #cv2.circle(image, (shape.part(point).x, shape.part(point).y), 1, (255, 255, 0), -1)
                landmark += [shape.part(point).x, shape.part(point).y]

        if dectfeature :
            # When using a distance threshold of 0.6, the dlib model obtains an accuracy
            # of 99.38% on the standard LFW face recognition benchmark, which is
            # comparable to other state-of-the-art methods for face recognition as of
            # February 2017. This accuracy means that, when presented with a pair of face
            # images, the tool will correctly identify if the pair belongs to the same
            # person or is from different people 99.38% of the time.

            #face_descriptor = facerec.compute_face_descriptor(image, shape, 100)
            face_descriptor += facerec.compute_face_descriptor(image, shape)
            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people.

    return facebox, landmark, face_descriptor



# 捕捉视频帧
lmain = tk.Label(imageFrame)
lmain.grid(row = 0, column = 0)
cap = cv2.VideoCapture(0)
#保存视频

# 视频的宽度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 视频的高度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 视频的编码
#fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) # not support YUYV
#fourcc = cv2.VideoWriter_fourcc(*'XVID') # not support XVID, but can work
fourcc = cv2.VideoWriter_fourcc(*"MPEG")# not support MPEG, but can work

# 定义视频输出
videoSave = cv2.VideoWriter("out.mp4", fourcc, fps, (width, height))


def show_frame():
    _, frame = cap.read()

    framenum = 0
    landmark = []
    feature = []
    if framenum % 33 == 0 :
        facebox,landmark,feature = dectAligRegFace(frame, dectfeature=True)

    framenum += 1

    blackimage = np.zeros((480, 640, 3), dtype=np.uint8)

    global gsetblackimage
    if gsetblackimage:
        frame = blackimage

    for point in range(0, len(landmark), 2):
        cv2.circle(frame, (landmark[point], landmark[point+1]), 3, (255, 255, 0), -1)

    global gBaseFaceFeather
    for i in range(0, len(facebox), 4):
        fect = i * 128
        vector1 = np.array(feature[fect:fect+128])
        vector2 = np.array(gBaseFaceFeather)
        dis = np.linalg.norm(vector1-vector2)
        if dis < 0.6 :
            print(dis)
            cv2.rectangle(frame, (facebox[i], facebox[i+1]), (facebox[i+2], facebox[i+3]), (0, 0, 255), 3)
        #    cv2.putText(image, 'dst {}'.format(i + 1), (d.left(), d.top() - 4), 0, 0.7, (0,0,255),1)
        #print(face_descriptor)

    #保存视频
    videoSave.write(frame)

    frame = cv2.flip(frame, 1)
    #转换成灰度图
    #cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image = img)
    lmain.imgtk = imgtk
    lmain.configure(image = imgtk)
    lmain.after(33, show_frame)


def setBlackImageFlag():
    global gsetblackimage
    gsetblackimage = not gsetblackimage


def drawFaceImg():
	faceBaseImgName = filedialog.askopenfilename()
	faceOrg = Image.open(faceBaseImgName)
	faceOrg = faceOrg.resize((200, 200))
	#faceOrg = ImageTk.PhotoImage(file = faceBaseImgName)
	face = ImageTk.PhotoImage(faceOrg)
	faceBaseImg = tk.Canvas(window)
	faceBaseImg.config(width = face.width(), height = face.height())
	faceBaseImg.grid(row = 1, column = 1, padx = 10, pady = 2)
	faceBaseImg.create_image(200, 200, image = face, anchor = tk.SE)

	baseFaceForDect = io.imread(faceBaseImgName)
	global gBaseFaceFeather
	_,_,gBaseFaceFeather = dectAligRegFace(baseFaceForDect, dectfeature=True)
	#print(gBaseFaceFeather)

	window.mainloop()
	window.quit()


# the target person lable
targetLable = tk.Label(window, text = '基准头像', font = 'Helvetica -12 bold', fg = 'green', bg = 'white')
# targetLable.config(width=20, height=1)
targetLable.grid(row = 0, column = 1, padx = 10, pady = 10)

#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width = 200, height = 200)
sliderFrame.grid(row = 1, column=1, padx=10, pady=2)


# 指定字体名称、大小、样式
#ft = tkFont.Font(family='Noyo Sans CJK SC', size=12, weight=tkFont.BOLD)

openImgButton = tk.Button(window, text = '打开基准头像',  fg = 'blue', command = drawFaceImg)
openImgButton.config(width = 10, height = 2)
openImgButton.grid(row = 2, column = 1, padx = 40, pady = 2)

openImgButton = tk.Button(window, text = '黑色背景',  fg = 'green', command = setBlackImageFlag)
openImgButton.config(width = 10, height = 2)
openImgButton.grid(row = 3, column = 1, padx = 40, pady = 2)

#recognition = tk.Button(window, text = '识别', fg = 'purple', command = dectface)
#recognition.config(width = 10, height = 2)
#recognition.grid(row = 4, column = 1, padx = 40, pady = 2)

closeWindow = tk.Button(window, text = '退出', fg = 'red', command = window.quit)
closeWindow.config(width = 10, height = 2)
closeWindow.grid(row = 4, column = 1, padx = 40, pady = 2)


show_frame()  #Display 2
window.mainloop()  #Starts GUI


# Release everything if job is finished
cap.release()
videoSave.release()
#cv2.destroyAllWindows()
