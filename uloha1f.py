import cv2
import numpy as np

cap = cv2.VideoCapture('IMG_0081.MOV')
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 50)

object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=100) #najst vhodny trashHold, varThreshold=40

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape

    # extract region of interest
    roi = frame[0:height, 0:width] # full scene

    mask = object_detector.apply(frame)
    _,mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

    contours, _  =cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        #calc are and remove small elements
        area=cv2.contourArea(cnt)
        if area > 100:
           # cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)


    #cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)
    cv2.imshow("roi",roi)

    key=cv2.waitKey(30)
    if key==27: #esc
        break
        cap.release()
        cv2.destroyAllWindows()