import FisherFace as FF
import numpy as np

faces_train, idLabel_train = FF.read_faces("face/train")
print(">>-1:ReadData: feaces.shape=", faces_train.shape, "\tidLabel.shape=", idLabel_train.shape)
K = 30
K1 = 90
# Task 1,3=========================enroll faces==========
# project all training images into corresponding feature space
W, LL, m = FF.myPCA(faces_train)
We = W[:, :K]
W1 = W[:, :K1]
print(">>0:PCA: We.shape=", We.shape, "\tLL.shape=", LL.shape, "\tm.shape=", m.shape)
print(">>0:LDA: W1.shape=", W1.shape)
proj_pca_train = np.dot(We.T, (faces_train.T - m).T).T
print(">>1:PCA-Projection: train.shape=", proj_pca_train.shape)
reduced_faces_lda = np.dot(W1.T, (faces_train.T - m).T)
print(">>1:LDA-ReduceDimension: train.shape=", reduced_faces_lda.shape)
W_lda, Centers, classLabels = FF.myLDA(reduced_faces_lda, idLabel_train)
proj_lda_train = np.dot(np.dot(W_lda.T, W1.T), (faces_train.T - m).T)
print(">>1:LDA-Projection: train.shape=", proj_lda_train.shape)
# compute the mean z of all his feature vectors
faces_mean_train = [[] for x in range(10)]
for label, vector in zip(idLabel_train, proj_pca_train):
	faces_mean_train[label].append(vector)
faces_mean_train = [np.mean(x, 0) for x in faces_mean_train]
faces_mean_train = np.asarray(faces_mean_train)
print(">>2:PCA-FacesMean: train.shape=", faces_mean_train.shape)
# store these vectors as columns in numpy-matrix Z for every person
Z = faces_mean_train.T
Z_lda = Centers
print(">>3:PCA- Z.shape=", Z.shape)
print(">>3:LDA- Z.shape=", Z_lda.shape)

# ======================identify========================
faces_test, idLabel_test = FF.read_faces("face/test")
proj_pca_test = np.dot(We.T, (faces_test.T - m).T)  # column for one sample
proj_lda_test = np.dot(np.dot(W_lda.T, W1.T), (faces_test.T - m).T)
print(">>4:PCA- Test-Projection: test.shape=", proj_pca_test.shape)
print(">>4:LDA- Test-Projection: test.shape=", proj_lda_test.shape)

# identify for PCA : compute distence
recog = []
for face in proj_pca_test.T:
	dist = [np.linalg.norm(temp - face) for temp in faces_mean_train]
	recog.append(dist.index(min(dist)))

# identify for LDA : compute distence
recog_lda = []
for face in proj_lda_test.T:
	dist = [np.linalg.norm(temp - face) for temp in Z_lda.T]
	recog_lda.append(dist.index(min(dist)))
# make up confusion matrix
confusionmatrix = [0 for x in range(10)]
confusionmatrix = np.asarray([confusionmatrix for x in range(10)])
confusionmatrix_lda = [0 for x in range(10)]
confusionmatrix_lda = np.asarray([confusionmatrix_lda for x in range(10)])

correct = 0
for recg, real in zip(recog, idLabel_test):
	if recg == real:
		correct += 1
	confusionmatrix[real][recg] += 1
print(">>4->5: PCA: ConfusionMatrix=\n", confusionmatrix)

correct_lda = 0
for recg, real in zip(recog_lda, idLabel_test):
	if recg == real:
		correct_lda += 1
	confusionmatrix_lda[real][recg] += 1
print(">>4->5: LDA: ConfusionMatrix=\n", confusionmatrix_lda)

print(">>5:PCA- OverallCorrectRate:", correct / len(recog), ",recgnized", correct, "of", len(recog))
print(">>5:LDA- OverallCorrectRate:", correct_lda / len(recog_lda), ",recgnized", correct_lda, "of", len(recog_lda))
# Task 5>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
alpha = 0.5
y = np.asarray([alpha * proj_pca_train.T, (1 - alpha) * proj_lda_train])
print(">>13:FusionScheme:  y.shape=", y.shape)
print(">>12:FusionScheme:  ye.shape=", y[0].shape, "\tyf.shape=", y[1].shape)
y = np.concatenate((y[0], y[1]), axis=0)
print(">>13:FusionScheme: Merged  y.shape=", y.shape)
# compute mean of every face
faces_mean_train_fusion = [[] for x in range(10)]
for label, vector in zip(idLabel_train, y.T):
	faces_mean_train_fusion[label].append(vector)
faces_mean_train_fusion = [np.mean(x, 0) for x in faces_mean_train_fusion]
faces_mean_train_fusion = np.asarray(faces_mean_train_fusion)
print(">>14:FusionScheme: FacesMean: train.shape=", faces_mean_train_fusion.shape)
# identify persoon
proj_test_fusion = np.concatenate((alpha * (np.dot(We.T, (faces_test.T - m).T)), \
								   (1 - alpha) * np.dot(np.dot(W_lda.T, W1.T), (faces_test.T - m).T)), axis=0)
print('>>TEST FUSION SHAPE: ', proj_test_fusion.shape)
recog_fusion = []
for face in proj_test_fusion.T:
	dist = [np.linalg.norm(temp - face) for temp in faces_mean_train_fusion]
	recog_fusion.append(dist.index(min(dist)))
# confusion matrix
confusionmatrix_fusion = [0 for x in range(10)]
confusionmatrix_fusion = np.asarray([confusionmatrix_fusion for x in range(10)])
correct_fusion = 0
for recg, real in zip(recog_fusion, idLabel_test):
	if recg == real:
		correct_fusion += 1
	confusionmatrix[real][recg] += 1
print(">>15:FusionScheme: ConfusionMatrix=\n", confusionmatrix)
print(">>16:FusionScheme: OverallCorrectRate:", correct_fusion / len(recog_fusion))
