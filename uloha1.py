import cv2
import numpy as np
from math import sqrt,pow


class carTracker():
    number = 0
    stacked_counter = 0
    tracker = None
    isAlive = False
    info = None
    box = (0, 0, 0, 0)
    old_box = box

    def __init__(self, id, inbox):
        self.number = id
        self.tracker = cv2.TrackerCSRT.create()
        #self.tracker = cv2.TrackerKCF.create()
        self.tracker.init(frame, inbox)
        self.box = inbox
        self.info = (str(self.number), "Blue")
        self.isAlive = True

    def update(self):
        if self.isAlive:
            self.old_box = self.box
            self.isAlive, self.box = self.tracker.update(frame)
        if self.isAlive:
            self.isAlive = (not self.isTrackerStacked()) & (not self.isRectOutOfFrame())

    def isTrackerStacked(self):
        if sizeBetween(self.box, self.old_box) < 0.5:
            self.stacked_counter = self.stacked_counter + 1
            if self.stacked_counter > 5:
                print("Tracker " + str(self.number) + " stacked!")
                return True
        else:
            self.stacked_counter = 0
        return False

    def isRectOutOfFrame(self):
        startX = self.box[0]
        startY = self.box[1]
        endX = self.box[0] + self.box[2]
        endY = self.box[1] + self.box[3]
        dimensions = frame.shape
        if startX > dimensions[1] | endX > dimensions[1] | startX < 0 | endX < 0:  # check X
            print("Out of frame in X for tracker " + str(self.number))
            return True
        if startY > dimensions[0] | endY > dimensions[0] | startY < 0 | endY < 0:  # check Y
            print("Out of frame in Y for tracker " + str(self.number))
            return True
        return False

    def draw(self):
        if self.isAlive:
            listbox = list(self.box)
            listbox[0] = listbox[0] - moveX
            listbox[1] = listbox[1] - moveY
            box = tuple(listbox)
            drawBoundingRect(frame, box, self.info, False)


def defineParams(hsvFrame, askedColor):
    # Set range for red color
    lower1Red = np.array([0, 70, 0])
    upper1Red = np.array([30, 255, 255])

    lower2Red = np.array([120, 0, 60])
    upper2Red = np.array([160, 130, 150])
    # Set range for green color
    green_lower = np.array([35, 40, 20], np.uint8)
    green_upper = np.array([90, 255, 255], np.uint8)
    # Set range for white color
    white_lower = np.array([0, 0, 175])
    white_upper = np.array([180, 100, 255])

    red_mask_1 = cv2.inRange(hsvFrame, lower1Red, upper1Red)
    red_mask_2 = cv2.inRange(hsvFrame, lower2Red, upper2Red)
    red_mask = red_mask_1 + red_mask_2
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
    blur = cv2.GaussianBlur(inputFrame, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    hsvFrame = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)
    mask = defineParams(hsvFrame, askedColor)

    res = cv2.bitwise_and(hsvFrame, hsvFrame, mask=mask)
    return res


def findColorPosInList(color):
    id = 0
    for c in colors:
        if c == color:
            return id
        id = id + 1
    return None

## draw bounding rect around contour (car)
## if it's called from tracker, just write number of car
def drawBoundingRect(img, box, textInfo, rect=True):
    # parse box
    x = box[0] + moveX
    y = box[1] + moveY
    w = box[2]
    h = box[3]

    # parse info
    text = textInfo[0]
    colorName = textInfo[1]
    colorIdx = findColorPosInList(colorName)
    if colorIdx is None:
        color = colors[1]
    else:
        color = colorsNum[colorIdx]
    #draw rectangle
    if rect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #draw text
    if rect:
        cv2.putText(img, text, (round(x + w / 4), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color)
    else:
        cv2.putText(img, text, (round(x + w / 4), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 0, 0))

## set where do we want to detect cars and possibly start tracking them
def setRegionOfInterest(frame):
    moveY = 230
    moveX = 650
    roi = frame[moveY:moveY+120, moveX:moveX+150]
    return roi, moveX, moveY

## find a color of current object passed -> from predefined colors (white, green, red)
## if it isn't any of them, return None as OTHER color
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

## determine if object could be detected after mask aplication
def isThisColorRight(maskedImg, expectedArea):
    cannyy = cv2.Canny(maskedImg, 125, 175)
    dilated = cv2.dilate(cannyy, (7, 7), iterations=3)
    #cv2.imshow("D",dilated)
    #cv2.waitKey(10000)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > expectedArea/2:
            return True
    return False

## find contours in img (original frame with applied background substractor)
## return valid contours with information about them: size of object (car), color
def findContours(img, original):
    # define 2 lists
    contoursList = []
    typesAndColors = []

    # first find contours
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # calc are and remove small elements
        area = cv2.contourArea(cnt)
        if area > 180:
            x, y, w, h = cv2.boundingRect(cnt)
            contour = (x, y, w, h)
            color = determineColorOfObject(original, contour, area)
            if area > 1000:
                type = "Big"
            else:
                type = "Small"
            if color is None:
                color = "Black"
            if isValidContour(contour, contoursList):
                contoursList.append(contour)
                typesAndColors.append((type, color))
    return contoursList, typesAndColors


#find if current contour is not in list yet
def isValidContour(contour, contoursList):
    for contourFromList in contoursList:
        if sizeBetween(contour, contourFromList) < 20:
            return False
    return True

#make sure we won't track already tracked object
#function return true if current found object is already in list of tracked objects, else will return false
def alreadyTrackedObject(tested_box):
    for tracker in trackers:
        if sizeBetween(tracker.box, tested_box) < 30:
            return True
    return False

## range between two boxes -> used to determine if this box is already shown/tracked
def sizeBetween(box1, box2):
    centerBox1 = findCenterOfBox(box1)
    centerBox2 = findCenterOfBox(box2)
    xdif = (centerBox1[0] - centerBox2[0])
    ydif = (centerBox1[1] - centerBox2[1])
    return sqrt(pow(xdif, 2) + pow(ydif, 2))


## find middle of box, used to range measurements
def findCenterOfBox(box):
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    center = (x, y)
    return center


## init new tracker from passed contour and info about contour
def initTracker(box, info):
    global numOfTrackedObjects
    colorName = info[1]
    nboxX = box[0] + moveX
    nboxY = box[1] + moveY
    nbox = (nboxX, nboxY, box[2], box[3])
    if colorName == "White" and not alreadyTrackedObject(nbox):
        numOfTrackedObjects = numOfTrackedObjects + 1
        tracker = carTracker(numOfTrackedObjects, nbox)
        trackers.append(tracker)
        print("Tracker initiated ", str(numOfTrackedObjects))

## destroy trackers that are tracking non - existing objects (gone out of camera)
def destroyUnactiveTrackers():
    global trackers
    idx = findUnactiveTrackers()
    for i in range(len(idx)):
        trackers.remove(trackers[idx[i]])

## find these unactive objects and return their position in list
def findUnactiveTrackers():
    idx = []
    for j in range(len(trackers)):
        unactiveTracker = not trackers[j].isAlive
        if unactiveTracker:
            idx.append(j)
    for i in range(1,len(idx)):
        idx[i] = idx[i] - i
    return idx


### START OF MAIN CODE

trackers = []               # global list of all trackers in frame
numOfTrackedObjects = 0     # number of tracked objects since start of program

colorsNum = [(250, 250, 250), (0,0,255), (0,255,0), (10, 10, 10)]   # BGR mode colors
colors = ["White", "Red", "Green", "Black"]                         # colors in strings -> for correct color it need to be in sync with BGR of colors
videos = ['rec_Trim.mp4', 'rec_Trim2.mp4']                          # name of videos (could be used with more)
choice = 0                                                          # choice of video

cap = cv2.VideoCapture(videos[choice])                              # start capture of video

# define background substractor
object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=100, varThreshold=60)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        break

    roi, moveX, moveY = setRegionOfInterest(frame)                          # set region of interest + remember move in X and Y
    mask = object_detector.apply(roi)                                       # apply object detector to ROI
    _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)              # treshold unwanted objects
    contours, contourInfo = findContours(mask, roi)                         # find contours in ROI

    if len(contours) != len(contourInfo):
        print("Not good")
        break

    for i in range(0, len(contours)):
        drawBoundingRect(frame, contours[i], contourInfo[i])                # bound found moving objects
        initTracker(contours[i], contourInfo[i])                            # try to find right object to track

    for tracker in trackers:
        tracker.update()                                                    # update trackers positions
        tracker.draw()                                                      # draw numbers of cars

    #cv2.imshow("roi", roi)                                                 #just for debug
    #maskedFramik = maskColor(frame, "Red")
    #cv2.imshow("masked frame", maskedFramik)
    cv2.imshow("Frame", frame)

    destroyUnactiveTrackers()

    key = cv2.waitKey(10)
    if key == 27:  # esc
        break
        cap.release()
        cv2.destroyAllWindows()

print("Num of cars: " + str(numOfTrackedObjects))