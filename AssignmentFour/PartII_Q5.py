#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
I do test THREE different factors which
may affect the results of face detector.
Factor A: number of faces in a single image.
Factor B: shadow(light) on the surface of face.
Factor C: the quality of image.
'''

import cv2

testA = 'NBA.jpg'  # to test the factor A.
testB = 'shadow.jpg'  # to test the factor B.
testC = 'blur.jpg'  # to test the factor C.


def detect_face(img_name):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# 'haarcascade_frontalface_default.xml' is provided by opencv, you can find it ine the opencv folder
	img = cv2.imread(img_name)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)
	if not len(faces):
		print('no face detected for', img_name)
	else:
		print(len(faces), 'faces detected for', img_name)

		for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

		cv2.imwrite(img_name[:-4] + '_result' + img_name[-4:], img)


if __name__ == '__main__':
	detect_face(testA)
	detect_face(testB)
	detect_face(testC)

'''
The results say that:

As for factor A(number of faces in a single image):
actually on the whole, it works well, but some faces were ignored, I guess 
that the differences between poses is the main reason.Cuz in shadow.jpg, there're 
also many faces, but the detector works quite well.

As for factor B(shadow(light) on the surface of face.):
except for the last one, which is badly lack of light, other heads were detected well.
So it means the shadow really does effects for the performence of face detectors.

As for factor C(the quality of image):
blur.jpg has kinds of faces in different quality levels.
and the result shows that faces in bad quality are usually undetected, 
but some exceptions also exist, which really confused me.
'''
