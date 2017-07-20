#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
:summary: this file includes script for Q3 [image process]
:author: CHEN QIUKAI
'''
import cv2
from matplotlib import pyplot as plt

file_name = 'bee.png'
src_img = cv2.imread(file_name)  # read the image

# Convert the image to HSV color space
cvted_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)

# Perform histogram equalization on V channel
print(cvted_img.shape)
cvted_img[:, :, 2] = cv2.equalizeHist(cvted_img[:, :, 2])
# draw the histogram
#plt.hist(cvted_img.flatten(), 'auto', histtype='stepfilled', color='b')
#plt.show()

# Convert the result image to BGR color space.
res_img = cv2.cvtColor(cvted_img, cv2.COLOR_HSV2BGR)

cv2.imshow('Result Image', res_img)
cv2.imwrite('Q3_ResultImage.png', res_img)  # save the image
cv2.waitKey(0)
cv2.destroyAllWindows()
