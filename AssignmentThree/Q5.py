#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from numpy import linalg

# read the flower
flower = Image.open('flower.bmp')
flower.show()

# convert it to grayscale
flower = ImageOps.grayscale(flower)

# convert image type to array
aflower = np.asarray(flower, dtype=np.uint8)
aflower = np.float32(aflower)

# compute the SVD
(U, S, Vt) = linalg.svd(aflower)

plt.plot(S, 'b.')
plt.savefig('SingularNumbers.jpg')
plt.show()

# the first few singular values are much bigger than the others, and they decrease rapidly.

# extract the first K singular values and their corresponding vectors in U and V
K = 20
Sk = np.diag(S[:K])
Uk = U[:, :K]
Vtk = Vt[:K, :]

# form the compressed image
aImk = np.dot(Uk, np.dot(Sk, Vtk))
Imk = Image.fromarray(aImk)
Imk.show()
Imk.convert('RGB').save('Flower_k_20.png', 'PNG')

# Print the compressed images for the four different values of K.
_, figs = plt.subplots(2, 2, sharey=True, sharex=True)
Ks = [20, 50, 200, 100]
for i in range(4):
	Sk = np.diag(S[:Ks[i]])
	Uk = U[:, :Ks[i]]
	Vtk = Vt[:Ks[i], :]
	if i < 2:
		figs[0, i].imshow(np.dot(Uk, np.dot(Sk, Vtk)), cmap='gray')
		figs[0, i].set_title('K={}'.format(Ks[i]))
	else:
		figs[1, i - 3].imshow(np.dot(Uk, np.dot(Sk, Vtk)), cmap='gray')
		figs[1, i - 3].set_title('K={}'.format(Ks[i]))

plt.savefig('Different_K.png')
plt.show()

# with the increasing of value K, the quality of image is also increasing.

# when the K increases to 100 , the image is clear enough, so there is no need to
# transmit when K=200, although it's much closer to the origin , it is more than wasting.
