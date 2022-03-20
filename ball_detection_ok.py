import cv2 as cv
import numpy as np
from collections import deque

videoCapture = cv.VideoCapture(0)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2
pts = deque(maxlen=64)
center = None

while True:
    ret, frame = videoCapture.read()
    if not ret: break
    
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blueFrame = frame[:,:,0]
    greenFrame = frame[:,:,1]
    redFrame = frame[:,:,2]
    blurFrame = cv.GaussianBlur(greenFrame, (0, 0), 8)
    
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100, param1 = 30, param2 = 60, minRadius= 5, maxRadius=200)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
                    center = chosen[0], chosen[1]
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)
        prevCircle = chosen
 
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)   
 
    cv.imshow('circles', frame)
    cv.imshow('green', greenFrame)
    cv.imshow('blur', blurFrame)

    
    if cv.waitKey(1) & 0xFF == ord('q'): break
    
videoCapture.release()
cv.destroyAllWindows()