import sys
import numpy as np
import cv2

skin_ycrcb_mint = np.array((80,133,77))
skin_ycrcb_maxt = np.array((255, 173, 127))

cap=cv2.VideoCapture('assets/cute little girl waving.mp4')
while cap.isOpened():
    _, frame=cap.read()
    im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    
    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(frame, contours, i, (255, 0, 0), 3)
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()        # Final image