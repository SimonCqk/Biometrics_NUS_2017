#!/usr/bin/python
# -*- coding:utf-8 -*-
import operator
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


def myLDA(A, Labels):
	# function [W,m]=myLDA(A,Label)
	# computes LDA of matrix A
	# A: D by N data matrix. Each column is a random vector
	# W: D by K matrix whose columns are the principal components in decreasing order
	# m: mean of each projection
	classLabels = np.unique(Labels)
	classNum = len(classLabels)
	dim, datanum = A.shape
	totalMean = np.mean(A, 1)
	partition = [np.where(Labels == label)[0] for label in classLabels]
	classMean = [(np.mean(A[:, idx], 1), len(idx)) for idx in partition]

	# compute the within-class scatter matrix
	W = np.zeros((dim, dim))
	for idx in partition:
		W += np.cov(A[:, idx], rowvar=1) * len(idx)

	# compute the between-class scatter matrix
	B = np.zeros((dim, dim))
	for mu, class_size in classMean:
		offset = mu - totalMean
		B += np.outer(offset, offset) * class_size

	# solve the generalized eigenvalue problem for discriminant directions
	ew, ev = linalg.eig(B, W)

	sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
	selected_ind = [ind for ind, val in sorted_pairs[:classNum - 1]]
	LDAW = ev[:, selected_ind]
	Centers = [np.dot(mu, LDAW) for mu, class_size in classMean]
	Centers = np.array(Centers).T
	return LDAW, Centers, classLabels


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


def PCA_enroll(train_path):
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


def PCA_identify(test_path, train_path):
	Z = PCA_enroll(train_path)
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
	misjudge = 0
	for idx in range(len(Y_PCA)):
		Z_cores = [np.linalg.norm(Z[i] - Y_PCA[idx], ord=2) for i in range(len(Z))]
		ans = Z_cores.index(min(Z_cores))
		if ans != (idx // 12):
			misjudge += 1
		if (idx + 1) % 12:
			print(ans, end=' ')
		else:
			print(ans)
	print('accuracy:{:.2f}%'.format((1 - misjudge / len(Y_PCA)) * 100))


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
	except:
		raise ImportError
	_, axes = plt.subplots(3, 3)
	for i in range(3):
		for j in range(3):
			if i == 2 and j == 2:
				break
			axes[i, j].imshow(re_faces[3 * i + j], cmap='gray')
			axes[i, j].set_title('Eigenface {}'.format(3 * i + j + 1))
	axes[2, 2].imshow(m, cmap='gray')
	axes[2, 2].set_title('mean')
	plt.subplots_adjust(hspace=0.5)
	plt.savefig('Eight_EigenFaces.png')
	plt.show()


def LDA_enroll(train_path):
	faces, id_label = read_faces(train_path)
	# train PCA
	(W, LL, m) = myPCA(faces)
	# LDA extract
	K1 = 90
	W1 = W[:, :K1]
	X_LDA = []
	for idx in range(faces.shape[1]):
		x = np.dot(W1.T, (faces.T[idx] - m))
		X_LDA.append(x)
	X_LDA = np.array(X_LDA).T
	# train LDA
	LDAW, centers, class_labels = myLDA(X_LDA, id_label)
	return centers


def LDA_identify(test_path, train_path):
	Z = LDA_enroll(train_path)
	Z = Z.T
	faces, id_label = read_faces(test_path)
	# train PCA
	(W, LL, m) = myPCA(faces)
	# LDA extract
	K1 = 90
	W1 = W[:, :K1]
	X_LDA = []
	for idx in range(faces.shape[1]):
		x = np.dot(W1.T, (faces.T[idx] - m))
		X_LDA.append(x)
	X_LDA = np.array(X_LDA).T
	# train LDA
	LDAW, _, _ = myLDA(X_LDA, id_label)
	Y_LDA = []
	for idx in range(faces.shape[1]):
		y = np.dot(np.dot(LDAW.T, W1.T), (faces[:, idx] - m))
		Y_LDA.append(y)
	misjudge = 0
	for idx in range(0, len(Y_LDA)):
		Z_cores = [np.linalg.norm(Y_LDA[idx] - Z[i], ord=2) for i in range(len(Z))]
		ans = Z_cores.index(min(Z_cores))
		if ans != (idx // 12):
			misjudge += 1
		if (idx + 1) % 12:
			print(ans, end=' ')
		else:
			print(ans)
	print('accuracy:{:.2f}%'.format((1 - misjudge / len(Y_LDA)) * 100))


def display_centers(train_path):
	faces, id_label = read_faces(train_path)
	# train PCA
	(W, LL, m) = myPCA(faces)
	# LDA extract
	K1 = 90
	W1 = W[:, :K1]
	centers, _, Wf = LDA_enroll(train_path)
	Cp = np.dot(Wf, centers)
	Cr = np.dot(W1, Cp) + np.tile(m, (10, 1)).T  # tile m to consist the dimension
	try:
		from matplotlib import pyplot as plt
	except:
		raise ImportError
	_, axes = plt.subplots(2, 5)
	Cr = Cr.T
	for i in range(2):
		for j in range(5):
			axes[i, j].imshow(Cr[2 * i + j].reshape(160, 140), cmap='gray')
			axes[i, j].set_title('Center {}'.format(2 * i + j))
	plt.show()


def fusion_identify(test_path, train_path):
	_, Y_PCA = PCA_enroll(train_path)
	_, Y_LDA, _ = LDA_enroll(train_path)
	Y_LDA = np.array(Y_LDA).astype(float)
	Y_PCA = np.array(Y_PCA).astype(float)
	alpha = 0.5
	Y_PCA, Y_LDA = np.multiply(alpha, Y_PCA), np.multiply((1 - alpha), Y_LDA)
	Y = np.append(Y_PCA, Y_LDA).reshape((39, 120)).T
	Z = []
	for i in range(0, len(Y), 12):
		z = Y[i:i + 12]
		Z.append(np.mean(z, axis=0))
	Z = np.array(Z)
	misjudge = 0
	for idx in range(0, len(Y)):
		Z_cores = [np.linalg.norm(Y[idx] - Z[i], ord=2) for i in range(len(Z))]
		ans = Z_cores.index(min(Z_cores))
		if ans != (idx // 12):
			misjudge += 1
		if (idx + 1) % 12:
			print(ans, end=' ')
		else:
			print(ans)
	print('accuracy:{:.2f}%'.format((1 - misjudge / len(Y_LDA)) * 100))


if __name__ == '__main__':
	# PCA_enroll(TRAIN_PATH)
	# PCA_identify(TEST_PATH, TRAIN_PATH)
	# display_eigenfaces(TRAIN_PATH)
	# LDA_enroll(TRAIN_PATH)
	LDA_identify(TEST_PATH, TRAIN_PATH)
# display_centers(TRAIN_PATH)
# fusion_identify(TEST_PATH, TRAIN_PATH)
