import cv2
import numpy as np

# Create tracker object

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

    if askedColor == 'r':
        mask = red_mask

    elif askedColor == 'w':
        mask = white_mask

    elif askedColor == 'g':
        mask = green_mask

    elif askedColor == 'b':
        mask = black_mask

    return mask


def maskColor(inputFrame, askedColor):

    kernal = np.ones((5, 5), "uint8")
        # 1. convert from BGR to HSV
    hsvFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
        #cv2.imshow("hsv", hsvFrame)

    mask = defineParams(hsvFrame, askedColor)
    #if askedColor == 'w':
    mask = cv2.dilate(mask, kernal)

        # 3.
    res = cv2.bitwise_and(hsvFrame, hsvFrame, mask=mask)

    masked = object_detector.apply(inputFrame)
    contours, _ = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return masked, contours, res


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


def findContours(contours, color, img):
            #   WHITE        ,    BLACK    ,     GREEN  ,    RED
    colors = [(250, 250, 250), (10, 10, 10), (0, 255, 0), (0, 0, 255)]

    colorOfText = colors[color]
    for cnt in contours:
        # calc are and remove small elements
        area = cv2.contourArea(cnt)
        if area > 2500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Big", (round(x + w / 4), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=colorOfText)
        elif area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Small", (round(x + w / 4), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=colorOfText)

### START OF CODE

videos = ['clasic.MOV', 'autobus.mp4','rec_Trim.mp4']
choice = 2
cap = cv2.VideoCapture(videos[choice])

object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=200, varThreshold=50)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = setRegionOfInterest(choice, frame)

    # create color masks
    whiteFrame, contoursWhite, rw = maskColor(roi, 'w')
    blackFrame, contoursBlack, rb = maskColor(roi, 'b')
    greenFrame, contoursGreen, rg = maskColor(roi, 'g')
    redFrame, contoursRed, rr = maskColor(roi, 'r')

    findContours(contoursWhite, 0, roi)
    findContours(contoursBlack, 1, roi)
    findContours(contoursGreen, 2, roi)
    findContours(contoursRed, 3, roi)

    # cv2.imshow("Frame",frame)
    cv2.imshow("black Mask", rb)
    cv2.imshow("green mask", rg)
    cv2.imshow("red mask", rr)
    cv2.imshow("white mask", rw)
    cv2.imshow("roi", roi)

#    print(color)
#    cv2.imshow("hsv maska", whiteFrame)
    key = cv2.waitKey(30)
    if key == 27:  # esc
        break
        cap.release()
        cv2.destroyAllWindows()
