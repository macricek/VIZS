import cv2
import numpy as np


def callback(x):
    pass

colors = ["White", "Red", "Green", "Black"]
videos = ['clasic.MOV', 'autobus.mp4', 'rec_Trim.mp4']
choice = 2
cap = cv2.VideoCapture(videos[choice])

cv2.namedWindow('image')

ilowH = 0
ihighH = 180

ilowS = 0
ihighS = 255
ilowV = 0
ihighV = 255

# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowH,180,callback)
cv2.createTrackbar('highH','image',ihighH,180,callback)

cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS,255,callback)

cv2.createTrackbar('lowV','image',ilowV,255,callback)
cv2.createTrackbar('highV','image',ihighV,255,callback)
ret, frame = cap.read()
while True:
    # grab the frame

    frame1 = frame
    if not ret:
        break
    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # show thresholded image
    cv2.imshow('image', frame)
    frame = frame1
    k = cv2.waitKey(10) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break
        cv2.destroyAllWindows()
        cap.release()
    elif k == 'p':
        k1 = cv2.waitKey(50000)
        if k1 == 'p':
            break