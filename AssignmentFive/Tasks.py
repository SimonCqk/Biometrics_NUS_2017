#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from FisherFace import myPCA, myLDA, read_faces, float2uint8

TRAIN_PATH = 'face/train'
TEST_PATH = 'face/test'
K = 30
K1 = 90


def PCA_enroll(train_path):
	faces, labels = read_faces(train_path)
	# train PCA
	(W, LL, m) = myPCA(faces)
	W_e = W[:, :K]
	# PCA extract
	Y_PCA = np.dot(W_e.T, (faces.T - m).T).T
	# compute mean feature matrix Z
	Z = [[] for i in range(10)]
	for label, z in zip(labels, Y_PCA):
		Z[label].append(z)
	Z = [np.mean(m, axis=0) for m in Z]
	Z = np.array(Z)
	return Z, m, W_e, Y_PCA.T, labels


def PCA_identify(test_path, train_path):
	Z, m, W_e, *_ = PCA_enroll(train_path)
	faces, labels = read_faces(test_path)
	# PCA extract for test faces
	Y_PCA = np.dot(W_e.T, (faces.T - m).T).T
	identified_PCA = []
	misjudge = 0
	for idx in range(len(Y_PCA)):
		# compute Euclidean distance
		Z_cores = [np.linalg.norm(z - Y_PCA[idx], ord=2) for z in Z]
		ans = Z_cores.index(min(Z_cores))
		identified_PCA.append(ans)
		if ans != (idx // 12):
			misjudge += 1
		'''
		if (idx + 1) % 12:
			print(ans, end=' ')
		else:
			print(ans)
		'''
	# create confusion matrix
	confusion = np.zeros((10, 10))
	for identify, origin in zip(identified_PCA, labels):
		confusion[origin, identify] += 1
	print('Confusion Matrix for PCA identify.\n', confusion)
	print('OverAll-Accuracy of PCA identify is:{:.2f}%'.format((1 - misjudge / len(Y_PCA)) * 100))


def display_eigenfaces(train_path):
	faces, labels = read_faces(train_path)
	(W, LL, m) = myPCA(faces)
	# extract 8 largest eigen-values
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
	W1 = W[:, :K1]
	X_LDA = np.dot(W1.T, (faces.T - m).T)
	# train LDA
	W_f, centers, class_labels = myLDA(X_LDA, id_label)
	# project face-matrix from PCA space to LDA space
	Y_LDA = np.dot(np.dot(W_f.T, W1.T), (faces.T - m).T)
	return centers, W_f, W1, m, Y_LDA


def LDA_identify(test_path, train_path):
	Z, W_f, W1, m, _ = LDA_enroll(train_path)
	Z = Z.T
	faces, labels = read_faces(test_path)
	# extract and project to LDA space
	Y_LDA = np.dot(np.dot(W_f.T, W1.T), (faces.T - m).T).T
	misjudge = 0
	identified_LDA = []
	for idx in range(0, len(Y_LDA)):
		# compute Euclidean distance
		Z_cores = [np.linalg.norm(Y_LDA[idx] - Z[i], ord=2) for i in range(len(Z))]
		ans = Z_cores.index(min(Z_cores))
		identified_LDA.append(ans)
		if ans != (idx // 12):
			misjudge += 1
			'''
		if (idx + 1) % 12:
			print(ans, end=' ')
		else:
			print(ans)
		'''
	# create confusion matrix
	confusion = np.zeros((10, 10))
	for identify, origin in zip(identified_LDA, labels):
		confusion[origin, identify] += 1
	print('Confusion Matrix for PCA identify.\n', confusion)
	print('OverAll-Accuracy of LDA identify is:{:.2f}%'.format((1 - misjudge / len(Y_LDA)) * 100))


def display_centers(train_path):
	faces, id_label = read_faces(train_path)
	Cf, Wf, W1, m, _ = LDA_enroll(train_path)
	Cp = np.dot(Wf, Cf)
	Cr = np.dot(W1, Cp) + np.tile(m, (10, 1)).T  # tile m to consist the dimension
	try:
		from matplotlib import pyplot as plt
	except:
		raise ImportError
	_, axes = plt.subplots(2, 5)
	Cr = Cr.T
	for i in range(axes.shape[0]):
		for j in range(axes.shape[1]):
			axes[i, j].imshow(Cr[5 * i + j].reshape(160, 140), cmap='gray')
			axes[i, j].set_title('Center {}'.format(5 * i + j + 1))
	plt.savefig('Ten_Centers.png')
	plt.show()


def fusion_identify(test_path, train_path, alpha=0.5):
	_, m, W_e, Y_PCA, labels = PCA_enroll(train_path)
	_, W_f, W1, _, Y_LDA = LDA_enroll(train_path)
	faces_test, labels_test = read_faces(test_path)
	# implement fusion schema to feature level (set alpha as 0.5)
	Y_fusion = np.concatenate((alpha * Y_PCA, (1 - alpha) * Y_LDA), axis=0)
	# compute mean matrix Z
	Z = [[] for i in range(10)]
	for label, z in zip(labels, Y_fusion.T):
		Z[label].append(z)
	Z = [np.mean(z, axis=0) for z in Z]
	Z = np.array(Z)
	# make up fused-feature matrix for faces
	Y_test = np.concatenate((alpha * (np.dot(W_e.T, (faces_test.T - m).T)), \
							 (1 - alpha) * np.dot(np.dot(W_f.T, W1.T), (faces_test.T - m).T)), axis=0)
	Y_test = Y_test.T
	identified_fusion = []
	misjudge = 0
	for idx in range(0, len(Y_test)):
		# compute Euclidean distance
		Z_cores = [np.linalg.norm(Y_test[idx] - z, ord=2) for z in Z]
		ans = Z_cores.index(min(Z_cores))
		identified_fusion.append(ans)
		if ans != (idx // 12):
			misjudge += 1
		'''
		if (idx + 1) % 12:
			print(ans, end=' ')
		else:
			print(ans)
		'''
	confusion = np.zeros((10, 10))
	for identify, origin in zip(identified_fusion, labels):
		confusion[origin, identify] += 1
	print('Confusion Matrix for Fusion Schema identify.\n', confusion)
	accuracy = 1 - misjudge / len(Y_test)
	print('OverAll-Accuracy of Fusion Schema [alpha={1}] identify is:{0:.2f}%'.format(accuracy * 100, alpha))
	return accuracy


def display_diff_alpha():
	alphas = [i / 10 for i in range(1, 10)]
	accuracies = [fusion_identify(TEST_PATH, TRAIN_PATH, alpha) for alpha in alphas]
	try:
		from matplotlib import pyplot as plt
	except:
		raise ImportError
	plt.plot(alphas, accuracies, 'bo')
	plt.title('Different Alpha Values for Fusion Schema')
	plt.xlabel('Alpha Values')
	plt.ylabel('Corresponding Accuracy')
	plt.savefig('DifferentAlphaValues.png')
	plt.show()


if __name__ == '__main__':
	PCA_identify(TEST_PATH, TRAIN_PATH)  # Task 1
	display_eigenfaces(TRAIN_PATH)  # Task 2
	LDA_identify(TEST_PATH, TRAIN_PATH)  # Task 3
	display_centers(TRAIN_PATH)  # Task 4
	fusion_identify(TEST_PATH, TRAIN_PATH)  # Task 5
	display_diff_alpha()
