import cv2

face_cascade=cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
eyes_cascade=cv2.CascadeClassifier('models/haarcascade_eye.xml')
cap=cv2.VideoCapture('assets/cute little girl waving.mp4')
while cap.isOpened():
    # read frame
    _, frame=cap.read()
    # convert frame to grayscale
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # detect faces
    faces=face_cascade.detectMultiScale(grayscale, 1.1,4)
    # detect smiles
    smiles=eyes_cascade.detectMultiScale(grayscale, 1.1,4)
    # draw rect frames
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),3)
    for(x,y,w,h) in smiles:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),3)
    # display output
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()