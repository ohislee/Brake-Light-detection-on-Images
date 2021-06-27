import cv2 as cv
import numpy as np

img = cv.imread('Photos/break_light1.jpg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

HSV_min = np.array([0,100,100])
HSV_max = np.array([179,255,255])

mask = cv.inRange(hsv, HSV_min, HSV_max)
result = cv.bitwise_and(img, img, mask= mask)

gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

contours, hierarchy = cv.findContours(gray_result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# ----------------to draw all contours
# for cnt in contours:
#     if cv.contourArea:
#         cv.drawContours(result, contours, -1, (0,255,0), 2)


# ------------to draw the maximum contour area
# max_contour = 0
# c = []
# for cnt in contours:
#     if cv.contourArea(cnt) > max_contour:
#         max_contour = cv.contourArea(cnt)
#         c = cnt

# cv.drawContours(result, [c], -1, (0,255,0), 2)


#-----------------to draw a bounding box around the contour area
max_contour = 0
brake_light = []
for cnt in contours:
    if cv.contourArea(cnt) > max_contour:
        max_contour = cv.contourArea(cnt)
        brake_light = cnt

x,y,w,h = cv.boundingRect(brake_light)
cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)

cv.imshow('Img', img)
cv.imshow('HSV', hsv)
cv.imshow('Mask', result)
cv.imshow('Gray', gray_result)


cv.imwrite('Photos/brake_light_detected1.jpg', img)

cv.waitKey(0)
