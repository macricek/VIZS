import cv2
import numpy as np


def defineParams(hsvFrame, askedColor):
    # Set range for red color
    red_lower = np.array([120, 60, 50], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    # Set range for green color
    green_lower = np.array([50, 60, 100], np.uint8)
    green_upper = np.array([120, 255, 255], np.uint8)
    # Set range for white color
    sensitivity = 70
    white_lower = np.array([0, 0, 255 - sensitivity])
    white_upper = np.array([255, sensitivity, 255])


    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    if askedColor == "Red":
        mask = red_mask

    elif askedColor == "White":
        mask = white_mask

    elif askedColor == "Green":
        mask = green_mask

    return mask


def maskColor(inputFrame, askedColor):
    kernal = np.ones((3, 3), "uint8")
    kernel = np.ones((5, 5), "uint8")
    blur = cv2.GaussianBlur(inputFrame, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
        # 1. convert from BGR to HSV
    hsvFrame = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)
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
    x = box[0] + moveX
    y = box[1] + moveY
    w = box[2]
    h = box[3]

    # parse info
    text = textInfo[0]
    colorName = textInfo[1]
    color = colorsNum[findColorPosInList(colorName)]
    if color is None:
        print("Picked black color!")
        color = colors[3]
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
        moveY = 230
        moveX = 650
        roi = frame[moveY:moveY+120, moveX:moveX+150]
    return roi, moveX, moveY


def isThisColorRight(img, expectedArea):

    #imgt = cv2.GaussianBlur(img,)

    cannyy = cv2.Canny(img, 125, 175)
    dilated = cv2.dilate(cannyy, (7, 7), iterations=3)
    #cv2.imshow("D",dilated)
    #cv2.waitKey(10000)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > expectedArea/2:
            return True
   # cv2.imshow("test", cannyy)
   # key = cv2.waitKey(2000)
    return False


def determineColorOfObject(img, roi, area):
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
    for currentColor in colors[:-1]:
        masked = maskColor(region, currentColor)
        if isThisColorRight(masked, area):
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
        if area > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            contour = (x, y, w, h)
            color = determineColorOfObject(original, contour, area)
            if area > 1000:
                type = "Big"
            else:
                type = "Small"
            if color is None:
                color = "Black"
            contoursList.append(contour)
            typesAndColors.append((type, color))
    return contoursList, typesAndColors

def initTracker(box, info):
    colorName = info[1]
    nboxX = box[0] + moveX
    nboxY = box[1] + moveY
    nbox = (nboxX, nboxY, box[2], box[3])
    if colorName == "White":
        tracker.init(frame, nbox)
        print("Tracker initiated")
        ninfo = ("Tracked car", "Black")
        bool = False
    else:
        bool = True
        ninfo = info
    return bool, nbox, ninfo


def updateTrackerStatus():
    logical, box = tracker.update(frame)
    if logical:
        if isRectOutOfFrame(box) or isTrackerStacked(box):
            return True, (0, 0, 0, 0)
    return not logical, box


def isRectOutOfFrame(box):
    startX = box[0]
    startY = box[1]
    endX = box[0] + box[2]
    endY = box[1] + box[3]
    dimensions = frame.shape
    if startX > dimensions[1] | endX > dimensions[1] | startX < 0 | endX < 0:   #check X
        print("Out of frame in X")
        return True
    if startY > dimensions[0] | endY > dimensions[0] | startY < 0 | endY < 0:   #check Y
        print("Out of frame in Y")
        return True
    return False


def isTrackerStacked(box):
    global stacked_counter
    if box == box_history:
        stacked_counter = stacked_counter + 1
        if stacked_counter > 5:
            print("Tracker stacked!")
            stacked_counter = 0
            return True
    return False

### START OF CODE
tracker = cv2.TrackerCSRT.create()
isTrackerFree = True
box_history = (0, 0, 0, 0)
stacked_counter = 0

colorsNum = [(250, 250, 250), (0, 255, 0), (0, 0, 255), (10, 10, 10)]
colors = ["White", "Red", "Green", "Black"]
videos = ['clasic.MOV', 'autobus.mp4', 'rec_Trim.mp4']
choice = 2
cap = cv2.VideoCapture(videos[choice])
ignoreFirst = 0
object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=100, varThreshold=50)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #b = cv2.selectROI("a", frame)
    #colormask = maskColor(frame, colors[1])
    #cv2.imshow("r",colormask)

    #key = cv2.waitKey(10000)
    if ignoreFirst > 3:
        roi, moveX, moveY = setRegionOfInterest(choice, frame)                  # set region of interest + remember move in X and Y
        mask = object_detector.apply(roi)                                       # apply object detector to ROI
        _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)              # treshold unwanted objects
        #cv2.imshow("maska", mask)
        contours, contourInfo = findContours(mask, roi)                         # find contours in ROI

        if len(contours) != len(contourInfo):
            print("Not good")

        for i in range(0, len(contours)):
            drawBoundingRect(frame, contours[i], contourInfo[i])                # bound found moving objects
            if isTrackerFree:
                isTrackerFree, box, info = initTracker(contours[i], contourInfo[i]) # if tracker is free, try to find right object to track

        if not isTrackerFree:
            isTrackerFree, box = updateTrackerStatus()
            listbox = list(box)
            box_history = tuple(listbox)
            listbox[0] = box[0] - moveX
            listbox[1] = box[1] - moveY
            box = tuple(listbox)
            drawBoundingRect(frame, box, info)
        cv2.imshow("roi", roi)
        cv2.imshow("Frame", frame)
    ignoreFirst = ignoreFirst + 1

#    print(color)
#    cv2.imshow("hsv maska", whiteFrame)
    key = cv2.waitKey(10)
    if key == 27:  # esc
        break
        cap.release()
        cv2.destroyAllWindows()
