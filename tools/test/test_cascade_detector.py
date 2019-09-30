# testing cascade classifiers for traffic signs detection
import time

import cv2 as cv

import utils.metrics as metrics
from utils.print import print_one_line

# args
cascade_path = "../../resourses/cascades/triang_rect_signs_cascade_detector.xml"
test_images_path = "D:/Downloads/traffic_signs_dataset/detection/test/images/"
test_annotations_path = "D:/Downloads/traffic_signs_dataset/detection/test/1/test2.txt"
min_size = (24, 24)
max_size = (144, 144)
iou = 0.5
show_false_negatives = False
show_all = False

test_annotations = open(test_annotations_path, "r").readlines()
detector = cv.CascadeClassifier(cascade_path)
test_counter = 1
for i in range(5):
    if i == 0:
        scale = 1.1
    if i == 1:
        scale = 1.15
    if i == 2:
        scale = 1.2
    if i == 3:
        scale = 1.25
    if i == 4:
        scale = 1.3

    for j in range(7):
        if j == 0:
            neighbors = 2
        if j == 1:
            neighbors = 3
        if j == 2:
            neighbors = 4
        if j == 3:
            neighbors = 5
        if j == 4:
            neighbors = 6
        if j == 5:
            neighbors = 7
        if j == 6:
            neighbors = 8

        images_len = len(test_annotations)
        images_count = 0
        real_objects_count = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        time_start = time.time()

        print("Test " + str(test_counter) + " : " + "scale=" + str(scale) + " neighbors=" + str(neighbors))
        for a in test_annotations:
            params = a.split(" ")
            fr = cv.imread(test_images_path + params[0])
            img = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
            detected = detector.detectMultiScale2(img, scale, minNeighbors=neighbors, minSize=min_size,
                                                  maxSize=max_size)
            detected_objects = detected[0]

            real_objects = []
            for k in range(int(params[1])):
                real_objects.append([int(params[2 + 4 * k]), int(params[3 + 4 * k]),
                                     int(params[4 + 4 * k]), int(params[5 + 4 * k])])

            real_objects_count += len(real_objects)
            c = 0
            for detected_obj in detected_objects:
                cv.rectangle(fr, (detected_obj[0], detected_obj[1]),
                             (detected_obj[0] + detected_obj[2], detected_obj[1] + detected_obj[3]),
                             (255, 0, 255), 2)

                for real_obj in real_objects:
                    k1 = metrics.iou_coeff(detected_obj, real_obj)
                    if k1 > iou:
                        true_positive += 1
                        c += 1

            for real_obj in real_objects:
                cv.rectangle(fr, (real_obj[0], real_obj[1]),
                             (real_obj[0] + real_obj[2], real_obj[1] + real_obj[3]),
                             (0, 255, 0), 2)

            false_positive += len(detected_objects) - c
            if c < len(real_objects):
                false_negative += len(real_objects) - c
                if show_false_negatives:
                    cv.imshow("False positives", fr)
                    if cv.waitKey(0) & 0xFF == ord('q'):
                        break
                    cv.destroyAllWindows()
            elif c > len(real_objects):
                true_positive -= c - len(real_objects)

            images_count += 1
            out = "%d of %d completed" % (images_count, images_len)
            print_one_line(out)

            if show_all:
                cv.imshow("Image", fr)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    break
                cv.destroyAllWindows()

        print("")

        precision = metrics.precision_metric(true_positive, false_positive)
        recall = metrics.recall_metric(true_positive, false_negative)
        f1score = metrics.f1score_metric(precision, recall)
        time_exec = (time.time() - time_start) / float(images_count)
        fps = round(1 / time_exec, 2)

        print("Frame time execution: " + str(time_exec))
        print("Fps: " + str(fps))
        print("Real positives: " + str(real_objects_count))
        print("True positives: " + str(true_positive))
        print("False positives: " + str(false_positive))
        print("False negatives: " + str(false_negative))
        print("Precision: " + str(round(precision * 100, 2)) + "%")
        print("Recall: " + str(round(recall * 100, 2)) + "%")
        print("f1-score: " + str(round(f1score * 100, 2)) + "%")
        print("-------------------------------------------------------------------------------------------------")
        test_counter += 1
