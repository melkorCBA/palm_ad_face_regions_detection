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
       

        def convolve(B, r):
            D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
            cv2.filter2D(B, -1, D, B)
            return B

        #Loading the image and converting to HSV
        image_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        model_hsv = image_hsv[225:400,400:700] # Select ROI

        #Get the model histogram M
        M = cv2.calcHist([model_hsv], channels=[0, 1], mask=None, 
                          histSize=[80, 256], ranges=[0, 180, 0, 256] )

        #Backprojection of our original image using the model histogram M
        B = cv2.calcBackProject([image_hsv], channels=[0,1], hist=M, 
                                 ranges=[0,180,0,256], scale=1)

        B = convolve(B, r=5)

        #Threshold to clean the image and merging to three-channels
        _, thresh = cv2.threshold(B, 30, 255, cv2.THRESH_BINARY)
        cv2.imshow("img1",cv2.cvtColor(model_hsv,cv2.COLOR_HSV2RGB))
        cv2.imshow("img2",cv2.bitwise_and(frame,frame, mask = thresh))

      
       
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
