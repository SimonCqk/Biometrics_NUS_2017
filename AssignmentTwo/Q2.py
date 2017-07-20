#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
:summary: this file includes functions for Q2 [image process]
:author: CHEN QIUKAI
'''
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import numpy as np

file_name = 'lena.png'
BOX = (100, 100, 300, 300)  # set default the origin vertice and offset value


def rotate_partly(file_name, box=BOX):  # Rotate a rectangular region
	with Image.open(file_name) as img:
		part = img.crop(box)
		part = part.rotate(45)  # default direction is counter-clockwise
		img.paste(part, box)  # paste the rotated image to the original one
		img.save('Q2_rotated_lena.png', 'PNG')
		img.show()


def draw_hist(file_name):  # Perform histogram equalization
	with Image.open(file_name) as img:
		img_arr = np.array(img)
		shape = img_arr.shape
		try:
			pcsed_img = Image.open('Q2_rotated_lena.png')  # load the processed image saved in ratate_partly()
		except:
			raise FileNotFoundError('No processed image found.')
		hist, bins = np.histogram(img_arr, bins=256)  # from 2D to 1D
		pcsed_hist = np.array(pcsed_img).flatten()  # from 2D to 1D
		# start to do histogram equalization for source image
		cdf = hist.cumsum()
		cdf_nor = cdf * 255 / cdf[-1]  # normalized
		# reshape the image
		img = np.interp(img_arr.flatten(), bins[:-1], cdf_nor)
		img = img.reshape(shape)
		# to show 2 histogram at the same time
		_, (axup, axdown) = plt.subplots(1, 2, sharey=True)
		# set titles & labels
		axup.set_title('Original Image Histogram')
		axdown.set_title('Processed Image Histogram')
		axup.set_xlabel('Pixel Counts')
		axdown.set_xlabel('Pixel Counts')
		# draw histograms in two subplots
		axup.hist(img.flatten(), 256, histtype='stepfilled', color='b')
		axdown.hist(pcsed_hist, 256, histtype='stepfilled', color='b')
		plt.savefig('Q2_Histograms.jpg', dpi=300)
		plt.show()


def filters(file_name):  # Perform filters
	with Image.open(file_name) as img:
		maxf = img.filter(ImageFilter.MaxFilter)
		minf = img.filter(ImageFilter.MinFilter)
		medianf = img.filter(ImageFilter.MedianFilter)
		maxf.save('Q2_MaxFilter.jpg', 'JPEG')
		minf.save('Q2_MinFilter.jpg', 'JPEG')
		medianf.save('Q2_MedianFilter.jpg', 'JPEG')
		maxf.show(title='Max Filter')
		minf.show(title='Min Filter')
		medianf.show(title='Median Filter')


def gau_blur(file_name):  # Perform Gaussian Blur
	with Image.open(file_name) as img:
		sigma3 = img.filter(ImageFilter.GaussianBlur(radius=3))
		sigma5 = img.filter(ImageFilter.GaussianBlur(radius=5))
		sigma3.save('Q2_Gaussian_Sigma_3.jpg', 'JPEG')
		sigma5.save('Q2_Gaussian_Sigma_5.jpg', 'JPEG')
		_, (left, right) = plt.subplots(1, 2, sharey=True)  # dispaly 2 images at the same time
		left.set_title('Sigma equals 3')
		right.set_title('Sigma equals 5')
		left.imshow(sigma3, cmap='gray')  # it's a gray scale image
		right.imshow(sigma5, cmap='gray')
		plt.show()


if __name__ == '__main__':
	rotate_partly(file_name)
	draw_hist(file_name)
	filters(file_name)
	gau_blur(file_name)
