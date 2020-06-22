import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

def nothing(x):
    pass


#load video
cap=cv2.VideoCapture('assets/cute little girl waving.mp4')
skin_min = np.array([0, 40, 150],np.uint8)
skin_max = np.array([20, 150, 255],np.uint8)  

 # total frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0  # holds current number

cv2.namedWindow("trackBar")
cv2.createTrackbar('area', "trackBar", 0, 12000, nothing)
cv2.createTrackbar('circleApproxVerticesMIN', "trackBar", 0, 50, nothing)
cv2.createTrackbar('circleApproxVerticesMAX', "trackBar", 0, 50, nothing)


while cap.isOpened():
    # read frame
    _, frame=cap.read()

    #original frame
    originalFrame=copy.deepcopy(frame)

    frame_number += 1


    #gaussian_blur
    gaussian_blur = cv2.GaussianBlur(frame,(5,5),0)
    #convert frame to gaussian_blu hsv
    blur_hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)

    #threshhold using min and max values
    tre_green = cv2.inRange(blur_hsv, skin_min, skin_max)
    # more erosion using 5*5 kernel
    kernel2 = np.ones((5, 5), np.uint8)
   

    #getting object green contour
    contours, hierarchy = cv2.findContours(tre_green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    mask = np.ones(tre_green .shape[:2], dtype="uint8") * 255
    #draw cotours

    #area from trackbar
    areaTrackbar=cv2.getTrackbarPos("area", "trackBar")
    minCircleVertices=cv2.getTrackbarPos('circleApproxVerticesMIN', "trackBar")
    maxCircleVertices=cv2.getTrackbarPos('circleApproxVerticesMAX', "trackBar")

    if(not areaTrackbar or areaTrackbar==0):
        cv2.setTrackbarPos("area", "trackBar", 10463)
        area=10463
    else:
        area=areaTrackbar


    if(not minCircleVertices or minCircleVertices==0):
        cv2.setTrackbarPos('circleApproxVerticesMIN', "trackBar", 8)
        minVertices=8
    else:
        minVertices=minCircleVertices

    if(not maxCircleVertices or maxCircleVertices==0):
        cv2.setTrackbarPos('circleApproxVerticesMAX', "trackBar", 23)
        maxVertices=23
    else:
        maxVertices=maxCircleVertices

    #hull points
    hull = []

    #initial contours
    # contourImage=copy.deepcopy(originalFrame)

     # find initial contours
    # Icontours, Ihierarchy = cv2.findContours(
    #         tre_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #draw initial contours
    # cv2.drawContours(contourImage, Icontours, -1, (255, 255, 255), 2)

    
    #head contours
    contour_list = []
    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour,0.02*cv2.arcLength(contour,True),True)
        areaCircle = cv2.contourArea(contour)
        if ((len(approx) > minVertices) & (len(approx) < maxVertices) & (areaCircle > area) ):
            contour_list.append(contour)
            #append to hull
            hull.append(cv2.convexHull(contour, False))
    #draw head 
    # cv2.drawContours(frame, contour_list,  -1, (0,255,0), 2)

    #circle contours 
    # contourImageWithCircleFilter=copy.deepcopy(originalFrame)
    # cv2.drawContours(contourImageWithCircleFilter, contour_list, -1, (255, 255, 255), 2)

    # draw ith convex hull object
    cv2.drawContours(frame, hull, -1, (255, 255, 255), 2)

    #find max filtered contour
    # maxContour=0
    # for k, cont in enumerate(contour_list):
    #     if(cv2.contourArea(cont)>=cv2.contourArea(contour_list[maxContour])):
    #         maxContour=k
    for j, cnt in enumerate(contour_list):
        x,y,w,h = cv2.boundingRect(cnt)
        text='detected'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        rect=cv2.minAreaRect(contour_list[j])
        box=cv2.boxPoints(rect)
        box=np.int0(box)
        cv2.drawContours(frame, [box], 0, (255,0,255),3)
    
    # for i, c in enumerate(contours):
    #     if cv2.contourArea(c)<area:
    #         x, y, w, h = cv2.boundingRect(c)
    #         cv2.drawContours(mask, [c], -1, (255), -1)
    #     else:
    #         # cv2.drawContours(frame, contours, i, (255, 0, 0), 3)
    #         rect=cv2.minAreaRect(contours[i])
    #         box=cv2.boxPoints(rect)
    #         box=np.int0(box)
    #         cv2.drawContours(frame, [box], 0, (255,0,0),3)

    displayFrame=cv2.bitwise_and(frame, frame, mask= mask)


    #draw contours
    # cv2.drawContours(frame,contours,-1,(0,255,0),3)
    
    # # draw rect frames
    # for(x,y,w,h) in faces:
    #     cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),3)

    #frame witing
    # if(frame_number==101):
    #     # cv2.imwrite('assets/outputFrames/gaussian_blur_frame101.jpg', gaussian_blur)
    #     # cv2.imwrite('assets/outputFrames/original_frame101.jpg', originalFrame)
    #     # cv2.imwrite('assets/outputFrames/binary_frame101.jpg', tre_green)
    #     #  cv2.imwrite('assets/outputFrames/init_contours_frame101.jpg', contourImage)
    #     # cv2.imwrite('assets/outputFrames/circle_contours_frame101.jpg', contourImageWithCircleFilter)
    #      cv2.imwrite('assets/outputFrames/final.jpg', frame)




    cv2.imshow("img",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()