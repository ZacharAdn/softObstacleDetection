import cv2
import numpy as np

# Cascade classifier for trees detection
treesCascade = cv2.CascadeClassifier('treesCascade.xml')
# Example video capture
videoCapture = cv2.VideoCapture('input/vid/cut_output2.mp4')

# Real-Time video capture
# videoCapture = cv2.VideoCapture(0)


# Find the obstacle marker
def find_marker(image, i):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 20, 400)  ##

    cv2.imwrite("distance_check/edge_%d.jpg" % i, edged)
    # cv2.imshow("Result",edged)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    print(c.shape)

    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)


# Calculate the distance to the obstacle
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


cm2inch = 2.54
cmDis = 600
KNOWN_DISTANCE = cmDis / cm2inch
i = 0
KNOWN_WIDTH = 20.0
exmpImage = cv2.imread("Result2.jpg")
marker = find_marker(exmpImage, i)
focalLength = (200 * KNOWN_DISTANCE) / KNOWN_WIDTH

if videoCapture.isOpened():
    rval, frame = videoCapture.read()
else:
    rval = False

# Video capture
while rval:
    rval, frame = videoCapture.read()

    # Trees detection
    trees = treesCascade.detectMultiScale(frame, 1.5, 2)

    ntrees = 0
    for (x, y, w, h) in trees:
        # Mark the tree and find his distance to camera
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        ntrees = ntrees + 1
        exmpImage = frame
        marker = find_marker(exmpImage, i)

        # print len(marker[])
        # cv2.imshow("Result",marker)

        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
        cm = inches * cm2inch

        # draw a bounding box around the image and display it
        box = np.int0(cv2.boxPoints((x, y, w, h)))
        cv2.drawContours(exmpImage, [box], -1, (0, 255, 0), 2)

        cv2.putText(exmpImage, "%.2f meter's" % (cm / 100),
                    (exmpImage.shape[1] - 400, exmpImage.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        # (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 3)


        cv2.imwrite("distance_check/image_out_%d.jpg" % i, exmpImage)

        if ntrees == 1:
            print(x, y, w, h)
            cv2.imwrite("Result2.jpg", cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2))

    # show result
    cv2.imshow("Result", frame)
    cv2.waitKey(1)
videoCapture.release()
