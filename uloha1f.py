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
    # Set range for BLACK color
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([255,255, 30])

    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    if askedColor == "Red":
        mask = red_mask

    elif askedColor == "White":
        mask = white_mask

    elif askedColor == "Green":
        mask = green_mask

    elif askedColor == "Black":
        mask = black_mask

    return mask


def maskColor(inputFrame, askedColor):
    kernal = np.ones((3, 3), "uint8")
        # 1. convert from BGR to HSV
    hsvFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
        #cv2.imshow("hsv", hsvFrame)
    mask = defineParams(hsvFrame, askedColor)
    mask = cv2.dilate(mask, kernal)
        # 3.
    res = cv2.bitwise_and(hsvFrame, hsvFrame, mask=mask)
    return res


def findColorPosInList(color):
    id = 0
    for c in colors:
        if c == color:
            return id
        id = id + 1
    return None


def drawBoundingRect(img, box, textInfo):
    # parse box
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    # parse info
    text = textInfo[0]
    colorName = textInfo[1]
    try:
        color = colorsNum[findColorPosInList(colorName)]
    except:
        color = colorsNum[0]
        print("Picked color!")
    #draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #draw text
    cv2.putText(img, text, (round(x + w / 4), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color)


def setRegionOfInterest(choice, frame):
    scale_percent = 60  # percent of original size
    if choice == 0:
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # extract region of interest
        roi = frame[300:600, 0:width]  # full scene
    elif choice == 1:
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # extract region of interest
        roi = frame[400:700, 0:width]  # full scene
    elif choice == 2:
        roi = frame[357:357+344, 680:680+547]
    return roi


def isThisColorRight():
    #TODO process masked image to determine
    return True


def determineColorOfObject(img, roi):
    trashHold = 10

    if roi[1] - trashHold > 0:
        h1 = roi[1] - trashHold
    else:
        h1 = roi[1]
    if roi[0] - trashHold > 0:
        w1 = roi[0] - trashHold
    else:
        w1 = roi[0]

    h2 = roi[1] + roi[3] + trashHold
    w2 = roi[0] + roi[2] + trashHold

    region = img[h1:h2, w1:w2]
    for currentColor in colors:
        masked = maskColor(region, currentColor)
        if isThisColorRight():
            return currentColor
    return None


def findContours(img, original):
    # define 2 lists
    contoursList = []
    typesAndColors = []

    # first find contours
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # calc are and remove small elements
        area = cv2.contourArea(cnt)

        if area > 4000:
            x, y, w, h = cv2.boundingRect(cnt)
            contour = (x, y, w, h)
            color = determineColorOfObject(original, contour)
            if area > 12000:
                type = "Big"
            else:
                type = "Small"
            contoursList.append(contour)
            typesAndColors.append((type, color))
    return contoursList, typesAndColors

### START OF CODE

colorsNum = [(250, 250, 250), (10, 10, 10), (0, 255, 0), (0, 0, 255)]
colors = ["White", "Black", "Green", "Red"]

videos = ['clasic.MOV', 'autobus.mp4', 'rec_Trim.mp4']
choice = 2
cap = cv2.VideoCapture(videos[choice])

object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=200, varThreshold=50)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = setRegionOfInterest(choice, frame)
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    contours, contourInfo = findContours(mask, roi)
    if len(contours) != len(contourInfo):
        print("Not good")
    for i in range(0, len(contours)):
        drawBoundingRect(roi, contours[i], contourInfo[i])


    # cv2.imshow("Frame",frame)
    #cv2.imshow("black Mask", rb)
    #cv2.imshow("green mask", rg)
    #cv2.imshow("red mask", rr)
    #cv2.imshow("white mask", rw)
    cv2.imshow("roi", roi)

#    print(color)
#    cv2.imshow("hsv maska", whiteFrame)
    key = cv2.waitKey(30)
    if key == 27:  # esc
        break
        cap.release()
        cv2.destroyAllWindows()
