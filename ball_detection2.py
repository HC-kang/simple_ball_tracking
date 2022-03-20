import cv2 as cv
import numpy as np

videoCapture = cv.VideoCapture(0)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2

while True:
    ret, frame = videoCapture.read()
    if not ret: break
    
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 8)
    
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100, param1 = 70, param2 = 55, minRadius= 5, maxRadius=400)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in range(circles.shape[1]):
            cx, cy, radius = circles[0][i]
            cv.circle(frame, (cx, cy), 1, (0, 100, 100), 3)
            cv.circle(frame, (cx, cy), radius, (255, 0, 255), 3)
            
    
    cv.imshow('circles', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'): break
    
videoCapture.release()
cv.destroyAllWindows()