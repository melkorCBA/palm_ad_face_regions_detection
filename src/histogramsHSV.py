# Required modules
import cv2
import numpy as np
import matplotlib.pyplot as plt

skin_min = np.array([0, 40, 150],np.uint8)
skin_max = np.array([20, 150, 255],np.uint8)  



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
        # Get pointer to video frames from primary device
        
        imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinRegionHSV = cv2.inRange(imageHSV, skin_min, skin_max)
        skinHSV = cv2.bitwise_and(frame, frame, mask = skinRegionHSV)

        frame_number += 1

        
        if(frame_number==101):
            color = ('r','g','b')
            labels=('H channel', 'S channel', 'V channel')
            for i,col in enumerate(color):
                histr = cv2.calcHist([skinHSV],[i],None,[256],[0,256])
                plt.plot(histr,color = col, label=labels[i])
                plt.legend(loc="upper left")
                plt.xlim([40,280])
                plt.ylim([0,1000])
                plt.title('Histogram for skin HSV levels')
            plt.show()
            cv2.imwrite('assets/outputFrames/maskedImage.jpg', skinHSV)

        cv2.imshow('img', skinHSV)
       
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
