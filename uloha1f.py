import cv2
import numpy as np

# Set range for red color
red_lower = np.array([136, 87, 111], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)
# Set range for green color
green_lower = np.array([25, 52, 72], np.uint8)
green_upper = np.array([102, 255, 255], np.uint8)
# Set range for blue color
blue_lower = np.array([94, 80, 2], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)
# Set range for white color
sensitivity = 40
white_lower = np.array([0,0,255-sensitivity])
white_upper = np.array([255,sensitivity,255])


def detectColor(inputFrame):
    kernal = np.ones((5, 5), "uint8")
    # 1. convert from BGR to HSV
    hsvFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv", hsvFrame)

    # 2. define color masks
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
    white_mask = cv2.dilate(white_mask, kernal)

    # 3.
    res = cv2.bitwise_and(hsvFrame, hsvFrame, mask=white_mask)
    #cv2.imshow("hsv maska", res)

    return "white", res



cap = cv2.VideoCapture('classic.MOV')
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(10, 50)

scale_percent = 60  # percent of original size

object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=100, varThreshold=90)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # extract region of interest
    roi = frame[300:600, 0:width]  # full scene
    color, coloredFrame = detectColor(roi)
    mask = object_detector.apply(coloredFrame)
    _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # calc are and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # cv2.imshow("Frame",frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("roi", roi)

    print(color)
    cv2.imshow("hsv maska", coloredFrame)
    key = cv2.waitKey(30)
    if key == 27:  # esc
        break
        cap.release()
        cv2.destroyAllWindows()
