# training hog+svm for traffic signs detection
import time

import cv2 as cv
import numpy as np

import utils.frameproc as frproc
from beans.traffic_signs import traffic_sign_hog_descriptor
from utils.print import print_one_line

# args
posTrainImagesPath = "D:/Downloads/traffic_signs_dataset/detection/train/positive_batch/images/"
posTrainAnnotationPath = "D:/Downloads/traffic_signs_dataset/detection/train/positive_batch/1/train.txt"
posCount = 8323  # 19170
negTrainImagesPath = "D:/Downloads/traffic_signs_dataset/detection/train/negative_batch/1/svm/images/"
negTrainAnnotationPath = "D:/Downloads/traffic_signs_dataset/detection/train/negative_batch/1/svm/neg.txt"
negCount = 10813  # 24044
svmModelSavePath = "D:/Downloads/traffic_signs_dataset/detection/train/positive_batch/1/svm/triang_rect_signs_hogsvm_detector_rbf_kfold10.xml"
kfold = 10

posTrainAnnotation = open(posTrainAnnotationPath, "r").readlines()
if posCount > len(posTrainAnnotation):
    posCount = len(posTrainAnnotation)

negTrainAnnotation = open(negTrainAnnotationPath, "r").readlines()
if negCount > len(negTrainAnnotation):
    negCount = len(negTrainAnnotation)

hog = traffic_sign_hog_descriptor()
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)

c = 0
samples = []
labels = []
print("Computing HOG descriptors for positives examples...")
for pos in posTrainAnnotation:
    if c < posCount:
        params = pos.split(" ")
        x1, y1 = int(params[2]), int(params[3])
        x2, y2 = x1 + int(params[4]), y1 + int(params[5])

        img = frproc.cut_frame(cv.imread(posTrainImagesPath + params[0].split("/")[1]), (x1, y1), (x2, y2))
        if (img.shape[1], img.shape[0]) != hog.winSize:
            img = cv.resize(img, hog.winSize, interpolation=cv.INTER_CUBIC)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        samples.append(hog.compute(img))
        labels.append(1)

        c += 1
        print_one_line("POS: %d computed of %d" % (c, posCount))
        del img
    else:
        break

c = 0
print("")
print("Computing HOG descriptors for negatives examples...")
for neg in negTrainAnnotation:
    if c < negCount:
        img = cv.imread(neg.replace("\n", ""))
        if (img.shape[1], img.shape[0]) != hog.winSize:
            img = cv.resize(img, hog.winSize, interpolation=cv.INTER_CUBIC)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        samples.append(hog.compute(img))
        labels.append(0)

        c += 1
        print_one_line("NEG: %d computed of %d" % (c, negCount))
        del img
    else:
        break

print("")
print("Feature vector length: " + str(len(samples[0])))
samples = np.float32(samples)
labels = np.array(labels)

rand = np.random.RandomState(666)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]

print("Training SVM...")
t = time.time()
svm.trainAuto(samples, cv.ml.ROW_SAMPLE, labels, kFold=kfold, balanced=True)
t = time.time() - t
minutes = t // 60
secs = int(t % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)
svm.save(svmModelSavePath)
print("Training SVM complete... Time: {} hours {} minutes {} seconds".format(hours, minutes, secs))
print("Model saved on path: " + svmModelSavePath)
