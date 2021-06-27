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
