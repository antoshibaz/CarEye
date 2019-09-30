# image cutter
import glob
import random as rnd

import cv2 as cv

import utils.frameproc as frproc
from utils.print import print_one_line

# args
inImagesPath = "D:/Downloads/traffic_signs_dataset/detection/train/negative_batch/1/svm/i/"
outImagesPath = "D:/Downloads/traffic_signs_dataset/detection/train/negative_batch/1/svm/images/"
cuttedImageSize = (24, 24)
cuttedImagesCount = 2000
nameNumber = 8067
genTxt = True

in_images = glob.glob1(inImagesPath, "*")
in_images_len = len(in_images)
cutted_in_image_count = (cuttedImagesCount // in_images_len) + 1
if cutted_in_image_count == 0:
    cutted_in_image_count = 1

c = 0
out_txt_lines = []
for i in in_images:
    if c == cuttedImagesCount:
        break

    img = cv.imread(inImagesPath + i)
    maybe_cutted_img_count = (img.shape[1] // cuttedImageSize[0]) * (img.shape[0] // cuttedImageSize[1])
    if maybe_cutted_img_count < cutted_in_image_count:
        cutted_in_image_count = maybe_cutted_img_count

    for k in range(cutted_in_image_count):
        p1 = (rnd.randint(0, img.shape[1] - cuttedImageSize[0] - 1),
              rnd.randint(0, img.shape[0] - cuttedImageSize[1] - 1))
        p2 = (p1[0] + cuttedImageSize[0], p1[1] + cuttedImageSize[1])
        s = outImagesPath + str(nameNumber) + "." + i.split(".")[1]
        cv.imwrite(s, frproc.cut_frame(img, p1, p2))
        c += 1
        nameNumber += 1

        if genTxt:
            out_txt_lines.append(s + "\n")

        print_one_line("%d of %d cutted" % (c, cuttedImagesCount))
        if c == cuttedImagesCount:
            break

if genTxt:
    open(outImagesPath + "annotations.txt", "w").writelines(out_txt_lines)
