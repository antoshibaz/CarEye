import cv2 as cv
import imutils as imu
import numpy as np

from utils import frameproc as frproc


# find contours and return rois for traffic signs
def contours_detector(frame, typeBinariz=0, binarizParams=(9, 2.5),
                      minSizeObj=(24, 24), maxSizeObj=(144, 144),
                      aspectRatioObjInterval=(0.6, 1.3), compactObjInterval=(0.01, 0.2)):
    # cut left and right parts
    fr_cut_main = cv.GaussianBlur(frame, (5, 5), 0)
    fr_cut_left = frproc.cut_frame(fr_cut_main, (0, 0), (int(0.3 * fr_cut_main.shape[1]), fr_cut_main.shape[0]))
    fr_cut_right = frproc.cut_frame(fr_cut_main, (int(0.7 * fr_cut_main.shape[1]), 0),
                                    (fr_cut_main.shape[1], fr_cut_main.shape[0]))

    # equalize histogram for left and right parts of frame
    fr_cut_left = cv.createCLAHE(2.0, (5, 5)).apply(fr_cut_left)
    fr_cut_left = cv.GaussianBlur(fr_cut_left, (5, 5), 0)
    fr_cut_right = cv.createCLAHE(2.0, (5, 5)).apply(fr_cut_right)
    fr_cut_right = cv.GaussianBlur(fr_cut_right, (5, 5), 0)

    # binarization type
    if typeBinariz == 0:
        tr_left = cv.adaptiveThreshold(fr_cut_left, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,
                                       binarizParams[0], binarizParams[1])
        tr_main = cv.adaptiveThreshold(fr_cut_main, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,
                                       binarizParams[0], binarizParams[1])
        tr_right = cv.adaptiveThreshold(fr_cut_right, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,
                                        binarizParams[0], binarizParams[1])
    elif typeBinariz == 1:
        tr_left = cv.Canny(fr_cut_left, binarizParams[0], binarizParams[1])
        tr_main = cv.Canny(fr_cut_main, binarizParams[0], binarizParams[1])
        tr_right = cv.Canny(fr_cut_right, binarizParams[0], binarizParams[1])
    elif typeBinariz == 2:
        tr_left = imu.auto_canny(fr_cut_left)
        tr_main = imu.auto_canny(fr_cut_main)
        tr_right = imu.auto_canny(fr_cut_right)
    else:
        return

    # concatenate parts in full frame
    tr_main[0:fr_cut_main.shape[0], 0:(int(0.3 * fr_cut_main.shape[1]))] = tr_left[::]
    tr_main[0:fr_cut_main.shape[0], int(0.7 * fr_cut_main.shape[1]):fr_cut_main.shape[1]] = tr_right[::]
    cv.imshow("Test contours", tr_main)

    _, contours, _ = cv.findContours(tr_main, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # sorting contours
    contours = [c for c in contours
                if maxSizeObj[0] >= cv.boundingRect(c)[2] >= minSizeObj[0]  # width
                and maxSizeObj[1] >= cv.boundingRect(c)[3] >= minSizeObj[1]  # height
                and (aspectRatioObjInterval[0] <=
                     (float(cv.boundingRect(c)[2]) / cv.boundingRect(c)[3]) <= aspectRatioObjInterval[1])  # aspect
                and ((cv.arcLength(c, True) != 0) and
                     (compactObjInterval[0] <=
                      (float(cv.contourArea(c)) / (cv.arcLength(c, True) ** 2)) <= compactObjInterval[1]))]  # compact

    rects_signs = [cv.boundingRect(c) for c in contours]
    return rects_signs


# find and return traffic signs rois through cascade classifier based on HAAR or LBP image descriptors
def cascade_detector(frame, cascadeDetector, slidingWindowScale, minNeighbors, minObjSize=(24, 24),
                     maxObjSize=(144, 144),
                     withWeightsPrediction=False):
    if not withWeightsPrediction:
        return cascadeDetector.detectMultiScale(frame, scaleFactor=slidingWindowScale, minNeighbors=minNeighbors,
                                                minSize=minObjSize, maxSize=maxObjSize)
    else:
        return cascadeDetector.detectMultiScale2(frame, scaleFactor=slidingWindowScale, minNeighbors=minNeighbors,
                                                 minSize=minObjSize, maxSize=maxObjSize)


# return prediction (sign or non-sign) for traffic signs through HOG image descriptor and SVM
# or find and return traffic signs rois in image
def hogsvm_detector(frame, svm, hog, mode=1):
    if mode:
        if (frame.shape[1], frame.shape[0]) != hog.winSize:
            frame = cv.resize(frame, hog.winSize, interpolation=cv.INTER_CUBIC)

        descriptor = np.float32([hog.compute(frame)])
        return svm.predict(descriptor)[1].ravel()[0], descriptor
    else:
        # implement
        return None
