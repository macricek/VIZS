import cv2
import numpy as np


def defineParams(hsvFrame, askedColor):
    # Set range for red color
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    # Set range for green color
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    # Set range for white color
    sensitivity = 50
    white_lower = np.array([0, 0, 255 - sensitivity])
    white_upper = np.array([255, sensitivity, 255])

    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
    if askedColor == 'r':
        mask = red_mask
        treshold = 120
    elif askedColor == 'w':
        mask = white_mask
        treshold = 120
    elif askedColor == 'g':
        mask = green_mask
        treshold = 120
    return mask,treshold


def maskColor(inputFrame, askedColor):

    if askedColor == 'b':
        grayscale = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("s", grayscale)
        _, res = cv2.threshold(grayscale, 1, 120, cv2.THRESH_BINARY)
    else:
        kernal = np.ones((5, 5), "uint8")
        # 1. convert from BGR to HSV
        hsvFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
        #cv2.imshow("hsv", hsvFrame)

        mask, treshold = defineParams(hsvFrame, askedColor)
        mask = cv2.dilate(mask, kernal)

        # 3.
        res = cv2.bitwise_and(hsvFrame, hsvFrame, mask=mask)

    masked = object_detector.apply(res)
    #_, masked = cv2.threshold(masked, treshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return masked, contours


def setRegionOfInterest(choice, frame):
    if choice == 0:
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # extract region of interest
        roi = frame[300:600, 0:width]  # full scene
    else:
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # extract region of interest
        roi = frame[400:700, 0:width]  # full scene

    return roi


def findContours(contours, color, img):
            #   WHITE        ,    BLUE    ,     GREEN  ,    RED
    colors = [(250, 250, 250), (0, 0, 0), (0, 255, 0), (0, 0, 255)]
    colorOfText = colors[color]
    for cnt in contours:
        # calc are and remove small elements
        area = cv2.contourArea(cnt)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "slavomir kajan", (round(x + w / 4), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=colorOfText)
        elif area > 250:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "kokot", (round(x + w / 4), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=colorOfText)

### START OF CODE

videos = ['clasic.MOV', 'autobus.mp4']
choice = 1
cap = cv2.VideoCapture(videos[choice])

scale_percent = 60  # percent of original size

object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=100, varThreshold=50)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = setRegionOfInterest(choice, frame)

    # create color masks
    whiteFrame, contoursWhite = maskColor(roi, 'w')
    blackFrame, contoursBlack = maskColor(roi, 'b')
    greenFrame, contoursGreen = maskColor(roi, 'g')
    redFrame, contoursRed = maskColor(roi, 'r')

    findContours(contoursWhite, 0, roi)
    findContours(contoursBlack, 1, roi)
    findContours(contoursGreen, 2, roi)
    findContours(contoursRed, 3, roi)

    # cv2.imshow("Frame",frame)
    cv2.imshow("Mask", blackFrame)
    cv2.imshow("roi", roi)

#    print(color)
#    cv2.imshow("hsv maska", whiteFrame)
    key = cv2.waitKey(30)
    if key == 27:  # esc
        break
        cap.release()
        cv2.destroyAllWindows()
