import operator
import os

import cv2
import numpy as np
import scipy.linalg as linalg


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
	ew, ev = linalg.eig(B, W + B)
	sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
	selected_ind = [ind for ind, val in sorted_pairs[:classNum - 1]]
	LDAW = ev[:, selected_ind]
	Centers = [np.dot(mu, LDAW) for mu, class_size in classMean]
	Centers = np.array(Centers).T
	return LDAW, Centers, classLabels


def myPCA(A):
	# function [W,LL,m]=mypca(A)
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
	print(order_index)
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
