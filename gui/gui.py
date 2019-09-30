import cv2 as cv
import numpy as np


def readTrafficSignImage(path1, path2):
    sign_image = cv.imread(path1)
    if sign_image is None:
        sign_image = cv.imread(path2)

    return sign_image


def showTrafficSignsOnLeftPanel(im, signsStack, imagesPath, isNewSign=False):
    pane_width = int(im.shape[1] * 0.33)
    im[0:im.shape[0], 0:pane_width] = np.array([255, 255, 255], dtype=int)

    if len(signsStack) != 0:
        # top sign
        top_sign_rect = [int(0.4 * pane_width) // 2, 48, int(0.6 * pane_width), int(0.6 * pane_width)]

        filename = signsStack.pop()
        sign_image = readTrafficSignImage(imagesPath + str(filename) + ".png",
                                          imagesPath + str(filename) + ".jpg")
        sign_image = cv.resize(sign_image, (top_sign_rect[2], top_sign_rect[3]))
        im[top_sign_rect[1]:top_sign_rect[1] + top_sign_rect[3],
        top_sign_rect[0]:top_sign_rect[0] + top_sign_rect[2]] = sign_image[:, :]

        previosly_sign_size = (int(0.25 * pane_width), int(0.25 * pane_width))

        if isNewSign:
            cv.putText(im, "NEW TRAFFIC SIGN", (34, (im.shape[0] // 2) - 12),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 4)
        else:
            cv.putText(im, "", (34, (im.shape[0] // 2) - 12),
                       cv.FONT_HERSHEY_SIMPLEX,
                       2, (255, 255, 255), 4)

        count = len(signsStack)
        start_x = int(0.1 * im.shape[0]) // 3
        xy = (start_x, (im.shape[0] // 2) + 64)
        for i in range(count):
            filename = signsStack.pop()
            sign_image = readTrafficSignImage(imagesPath + str(filename) + ".png",
                                              imagesPath + str(filename) + ".jpg")
            sign_image = cv.resize(sign_image, (previosly_sign_size[0], previosly_sign_size[1]))
            im[xy[1]:xy[1] + previosly_sign_size[1],
            xy[0]:xy[0] + previosly_sign_size[0]] = sign_image[:, :]

            if i == 2:
                xy = (start_x, xy[1] + previosly_sign_size[1] + 48)
            else:
                xy = (xy[0] + previosly_sign_size[0] + start_x, xy[1])
