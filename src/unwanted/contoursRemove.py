
import sys
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt


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

    # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
    #     'M', 'J', 'P', 'G'), 10, (int(cap.get(3)), int(cap.get(4))))

    # total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        _, frame = cap.read()
        frame_number += 1

        im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

        binaryImage = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
        kernel2 = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(binaryImage, kernel2, iterations=2)

        closing = cv2.morphologyEx(
            erosion, cv2.MORPH_CLOSE, kernel2, iterations=4)
        opening = cv2.morphologyEx(
            closing, cv2.MORPH_OPEN, kernel2, iterations=1)

        #find contours
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.ones(opening.shape[:2], dtype="uint8") * 255

        #draw cotours
        if (frame_number<2000):
            for i, c in enumerate(contours):
                if cv2.contourArea(c)>10000:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.drawContours(mask, [c], 0, (255), -1)

                else:
                    cv2.drawContours(frame, contours, i, (255, 0, 0), 3)

                    
        displayFrame=cv2.bitwise_and(frame, frame, mask= mask)

        cv2.putText(displayFrame, 'frame number: '+str(frame_number) +
                    "/"+str(total_frames), (50, 50), font, 0.8, (255, 0, 0), 2)
        cv2.imshow('img',displayFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cv2.destroyAllWindows() 
    cap.release()

def is_contour_bad(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    return len(approx), peri

def print_contour_bad(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    print("arcLength:"+str(peri))
    print("approx: "+ str(len(approx)))




main()






