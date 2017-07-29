#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

import cv2
import numpy as np
import scipy.linalg as linalg

TRAIN_PATH = 'face/train'
TEST_PATH = 'face/test'


def ComputeNorm(x):
	# function r=ComputeNorm(x)
	# computes vector norms of x
	# x: d x m matrix, each column a vector
	# r: 1 x m matrix, each the corresponding norm (L2)

	[row, col] = x.shape
	r = np.zeros((1, col))

	for i in range(col):
		r[0, i] = linalg.norm(x[:, i])
	return r


def myPCA(A):
	# function [W,LL,m]=myPCA(A)
	# computes PCA of matrix A
	# A: D by N data matrix. Each column is a random vector
	# W: D by K matrix whose columns are the principal components in decreasing order
	# LL: eigenvalues
	# m: mean of columns of A

	# Note: "lambda" is a Python reserved word


	# compute mean, and subtract mean from every column
	[r, c] = A.shape
	m = np.mean(A, 1)
	A = A - np.tile(m, (c, 1)).T
	B = np.dot(A.T, A)
	[d, v] = linalg.eig(B)

	# sort d in descending order
	order_index = np.argsort(d)
	order_index = order_index[::-1]
	d = d[order_index]
	v = v[:, order_index]

	# compute eigenvectors of scatter matrix
	W = np.dot(A, v)
	Wnorm = ComputeNorm(W)

	W1 = np.tile(Wnorm, (r, 1))
	W2 = W / W1

	LL = d[0:-1]

	W = W2[:, 0:-1]  # omit last column, which is the nullspace

	return W, LL, m


def read_faces(directory):
	# function faces = read_faces(directory)
	# Browse the directory, read image files and store faces in a matrix
	# faces: face matrix in which each colummn is a colummn vector for 1 face image
	# idLabels: corresponding ids for face matrix

	A = []  # A will store list of image vectors
	Label = []  # Label will store list of identity label

	# browsing the directory
	for f in os.listdir(directory):
		if not f[-3:] == 'bmp':
			continue
		infile = os.path.join(directory, f)
		im = cv2.imread(infile, 0)
		# turn an array into vector
		im_vec = np.reshape(im, -1)
		A.append(im_vec)
		name = f.split('_')[0][-1]
		Label.append(int(name))

	faces = np.array(A, dtype=np.float32)
	faces = faces.T
	idLabel = np.array(Label)

	return faces, idLabel


def float2uint8(arr):
	mmin = arr.min()
	mmax = arr.max()
	arr = (arr - mmin) / (mmax - mmin) * 255
	arr = np.uint8(arr)
	return arr


def enroll(train_path):
	faces, labels = read_faces(train_path)
	# train PCA
	(W, LL, m) = myPCA(faces)
	K = 30
	W_e = W[:, :K]
	# PCA extract
	Y_PCA = []
	for idx in range(faces.shape[1]):
		y = np.dot(W_e.T, (faces.T[idx] - m))
		Y_PCA.append(y)
	# compute matrix Z
	Z = []
	for i in range(0, len(Y_PCA), 12):
		z = Y_PCA[i:i + 12]
		Z.append(np.mean(np.transpose(z), axis=1))
	Z = np.array(Z)
	return Z


def identify(test_path, train_path):
	faces, labels = read_faces(test_path)
	# train PCA
	(W, LL, m) = myPCA(faces)
	K = 30
	W_e = W[:, :K]
	# PCA extract
	Y_PCA = []
	for idx in range(faces.shape[1]):
		y = np.dot(W_e.T, (faces.T[idx] - m))
		Y_PCA.append(y)
	Z = enroll(train_path)
	for idx in range(len(Y_PCA)):
		Z_cores = [np.linalg.norm(Y_PCA[idx] - Z[i], ord=2) for i in range(len(Z))]
		ans = Z_cores.index(min(Z_cores))
		if (idx + 1) % 10:
			print(ans, end=' ')
		else:
			print(ans)


def display_eigenfaces(train_path):
	faces, labels = read_faces(train_path)
	(W, LL, m) = myPCA(faces)
	eigenfaces = W[:, :8]
	re_faces = [face.reshape(160, 140) for face in eigenfaces.T]  # reshape faces
	m = m.reshape(160, 140)  # reshape mean vector
	for item in (re_faces + m):
		item = float2uint8(item)
	try:
		from matplotlib import pyplot as plt
	except ImportError:
		pass
	_, axes = plt.subplots(3, 3)
	for i in range(3):
		for j in range(3):
			if i == 2 and j == 2:
				break
			axes[i, j].imshow(re_faces[3 * i + j], cmap='gray')
			axes[i, j].set_title('Eigenface {}'.format(3 * i + j + 1))
	axes[2, 2].imshow(m, cmap='gray')
	axes[2, 2].set_title('mean')
	plt.savefig('Eight_EigenFaces.png')
	plt.show()


if __name__ == '__main__':
	enroll(TRAIN_PATH)
	identify(TEST_PATH, TRAIN_PATH)
# display_eigenfaces(TRAIN_PATH)
