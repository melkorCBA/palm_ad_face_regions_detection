import sys
import numpy as np
import cv2
import copy


class Enums:
    posValue = 255
    negValue = 0
    kernelSize = int(4)
    zero = 0
    full = kernelSize*kernelSize*posValue


# paper Face Segmentation Using Skin-Color Map in Videophone Applications by Chai and Ngan
skin_ycrcb_mint = np.array((80, 133, 77))
skin_ycrcb_maxt = np.array((255, 173, 127))


def main():

    font = cv2.FONT_HERSHEY_SIMPLEX  # font
    frame_number = 0  # holds current number

    # load video file
    cap = cv2.VideoCapture('assets/cute little girl waving.mp4')

    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (int(cap.get(3)), int(cap.get(4))))

    # total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        _, frame = cap.read()

        # convert to ycrcb
        im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

        # apply color segmentation
        # binary grayscale
        # skin colour 255, non-skin colour 0
        # binary  image
        # binaryImage = convertBinary(
        #     im_ycrcb, Enums.posValue, Enums.negValue, skin_ycrcb_mint, skin_ycrcb_maxt)

        binaryImage = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
        # more erosion using 5*5 kernel
        kernel2 = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(binaryImage, kernel2, iterations=2)

        closing = cv2.morphologyEx(
            erosion, cv2.MORPH_CLOSE, kernel2, iterations=4)
        opening = cv2.morphologyEx(
            closing, cv2.MORPH_OPEN, kernel2, iterations=1)

        # # findcontours
        # contours, hierarchy = cv2.findContours(
        #     opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # displayFrame = copy.deepcopy(frame)

        # # draw contours
        # for i, c in enumerate(contours):
        #     # M = cv2.moments(c)
        #     # # print(M)
        #     area = cv2.contourArea(c)
        #     if area > 15000:
        #         rect=cv2.minAreaRect(c)
        #         box=cv2.boxPoints(rect)
        #         box=np.int0(box)
                
        #         #moments
        #         W = rect[1][0]
        #         H = rect[1][1]

        #         Xs = [i[0] for i in box]
        #         Ys = [i[1] for i in box]
        #         x1 = min(Xs)
        #         x2 = max(Xs)
        #         y1 = min(Ys)
        #         y2 = max(Ys)
        #         headRatio1=W*1.5
        #         headRatio2=H-W*1.5

        #         #head boundary
                

        #         # cv2.drawContours(displayFrame, contours, i, (255, 0, 0), 3)
        #         cv2.drawContours(displayFrame, [box], 0, (255,0,0),3)
        # distanceTransformImage = np.zeros((opening.shape[0], opening.shape[1]), np.uint8)
        distanceTransformImage=cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        displayFrame = copy.deepcopy(distanceTransformImage)
        

        # developer info----
        frame_number += 1  # increment frame number
        cv2.putText(displayFrame, 'frame number: '+str(frame_number) +
                    "/"+str(total_frames), (50, 50), font, 0.8, (255, 0, 0), 2)
        # developer info----

        # write frame
        # out.write(displayFrame)

        # show frame
        cv2.imshow('img', displayFrame)
        # print(displayFrame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()


def convertBinary(image, posValue, negValue, minCValue, maxCValue):
    binaryImage = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            binaryImage[y, x] = negValue if(False in np.greater(
                image[y, x], minCValue) or False in np.less(image[y, x], maxCValue)) else posValue
    return binaryImage





main()
