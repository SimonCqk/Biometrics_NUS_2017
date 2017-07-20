#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
:summary: this file includes scripts for Q4 [image process of color changing]
:author: CHEN QIUKAI
'''
import cv2

# read the image
file_name = 'bee.png'
img = cv2.imread(file_name)

# convert the image to HSV
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# show and save mask
mask = cv2.inRange(img, (25, 0, 0), (35, 255, 255))
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.imwrite('Q4_mask.png', mask)
cv2.destroyAllWindows()

# Keep the background region in H channel.
H_bg = cv2.bitwise_and(img[:, :, 0], 255 - mask)
# Change the flower color to Magenta, whose H value is 150.
img[:, :, 0] = img[:, :, 0] + 130
H_roi = cv2.bitwise_and(img[:, :, 0], mask)
# Combine background and new flower in H channel.
H = cv2.bitwise_or(H_bg, H_roi)

_, S, V = cv2.split(img)
new_img = cv2.merge([H, S, V])

new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
cv2.imshow('the new image', new_img)
cv2.imwrite('Q4_NewImage.jpg', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
