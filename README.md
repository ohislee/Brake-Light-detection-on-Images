In the world of Autonomous vehicle the need for additional intelligence cannot be overemphasized.
The goal of this project is to use computer vision to build a vehicle brake light detection system on images.
This would later be extended to videos and real life applications. 

Looking at vehicle brake light images, what one sees are dispersed red colors coming from a light fitting of different shapes and sizes.
I would be focusing on the dispersed red lights and not the light fitting to build this detection system.

In building the detection system, the image is first, converted to HSV.
The HSV_min and HSV_max that works for the brake light are gotten through a different set of codes.
Masked HSV image is converted to gray image.
Contours for the gray image is extracted and then converted to a rectangular bounding box.
Bounding box with the maximum area is retained as the best detector for the brake light.
![Brake light detected images-1](https://user-images.githubusercontent.com/71301809/123559860-8fa5b500-d796-11eb-9165-2727323af036.jpg)



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


cv.waitKey(0)
