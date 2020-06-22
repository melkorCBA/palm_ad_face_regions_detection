
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

        # im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

           #gaussian_blur
        gaussian_blur = cv2.GaussianBlur(frame,(5,5),0)
        #convert frame to gaussian_blu hsv
        blur_hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)


        
        if(frame_number==5 or frame_number==500 or frame_number==1000):
            drawHist(blur_hsv, 0, plt, "yellow", "H hist")
            drawHist(blur_hsv, 1, plt, "blue", "S hist")
            drawHist(blur_hsv, 2, plt, "red", "v hist")
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def drawHist(image, chanel, plt, color, label):
    valuesChanel=[]
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
                valuesChanel.append(image[x, y,chanel])
    plt.hist(valuesChanel, bins=range(256), color=color, label=label)
    plt.show()
    return



main()






