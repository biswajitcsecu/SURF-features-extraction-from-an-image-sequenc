
import sys
import glob as gb
import os
import numpy as np
import cv2 as cv



imgs=gb.glob('image/*.jpg')

for fname in imgs:

 img = cv.imread(fname, cv.IMREAD_COLOR)
 gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

 sift = cv.xfeatures2d.SIFT_create()
 kp = sift.detect(gray,None)

 img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 cv.namedWindow('sift_keypoints')
 cv.imshow('sift_keypoints',img)


 cv.waitKey(2500)

cv.destroyAllWindows()
