import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

#cascades
face_cascade=cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')

#load video
cap=cv2.VideoCapture('assets/cute little girl waving.mp4')
skin_min = np.array([0, 40, 150],np.uint8)
skin_max = np.array([20, 150, 255],np.uint8)  


while cap.isOpened():
    # read frame
    _, frame=cap.read()
    #gaussian_blur
    gaussian_blur = cv2.GaussianBlur(frame,(5,5),0)
    #convert frame to gaussian_blu hsv
    blur_hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)

    # # convert frame to grayscale
    # grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # # detect faces
    # faces=face_cascade.detectMultiScale(grayscale, 1.1,4)

    #threshhold using min and max values
    tre_green = cv2.inRange(blur_hsv, skin_min, skin_max)
    #getting object green contour
    contours, hierarchy = cv2.findContours(tre_green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #draw contours
    cv2.drawContours(frame,contours,-1,(0,255,0),3)
    
    # # draw rect frames
    # for(x,y,w,h) in faces:
    #     cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),3)

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()