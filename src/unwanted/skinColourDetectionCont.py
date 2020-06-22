import sys
import numpy as np
import cv2

#order
#1.colour segmentation
#2.Density Regularization
#3.Luminance Regularization
#4.Geometric Correction
#5.Contour Extraction
#
#


#paper Face Segmentation Using Skin-Color Map in Videophone Applications by Chai and Ngan
skin_ycrcb_mint = np.array((80,133,77))
skin_ycrcb_maxt = np.array((255, 173, 127))

#font
font = cv2.FONT_HERSHEY_SIMPLEX 
frame_number=0

cap=cv2.VideoCapture('assets/cute little girl waving.mp4')
total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while cap.isOpened():
    _, frame=cap.read()
    cv2.putText(frame, 'frame number: '+str(frame_number)+"/"+str(total_frames), (50, 50), font, 0.8, (255, 0, 0), 2)
    im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
    #can apply #gaussian_blur for mor segmentation 

    #apply colour segmentation
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    #skin colour 0, non-skin colour 1

    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(frame, contours, i, (255, 0, 0), 3)
    cv2.imshow('img',frame)
    #increment frame number
    frame_number+=1
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()        # Final image