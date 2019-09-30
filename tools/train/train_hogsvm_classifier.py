# training hog+svm for traffic signs classification
import glob
import os
import time

import cv2 as cv
import numpy as np

from beans.traffic_signs import traffic_sign_hog_descriptor
from utils.print import print_one_line

# args
trainPath = "D:/Downloads/traffic_signs_dataset/classification/train"
testPath = "D:/Downloads/traffic_signs_dataset/classification/test"
svmModelSavePath = "D:/Downloads/traffic_signs_dataset/classification/signs_classifier_lin_bal.xml"
kfold = 10
show_false_negatives = False

hog = traffic_sign_hog_descriptor()
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)

classes = [c[0] for c in os.walk(trainPath)]
classes = classes[1:]
classes = [int(c.split("\\")[1]) for c in classes]
classes.sort()
print(classes)

labels = []
samples = []
print("Computing HOG descriptors for samples...")
for cl in classes:
    trainPathCl = trainPath + "/" + str(cl)
    imgs = glob.glob1(trainPathCl, "*")
    c = 0
    for i in imgs:
        img = cv.cvtColor(cv.imread(trainPathCl + "/" + i), cv.COLOR_BGR2GRAY)
        samples.append(hog.compute(img))
        labels.append(cl)
        c += 1
        print_one_line("For class %d: %d computed of %d" % (cl, c, len(imgs)))

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
svm.trainAuto(samples, cv.ml.ROW_SAMPLE, labels, balanced=True)
t = time.time() - t
minutes = t // 60
secs = int(t % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)
svm.save(svmModelSavePath)
print("Training SVM complete... Time: {} hours {} minutes {} seconds".format(hours, minutes, secs))
print("Model saved on path: " + svmModelSavePath)
