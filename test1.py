import cv2
import numpy as np
import math

img = cv2.imread('test_images/img_1330.jpg', 0)
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
_, contours, __ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('before contours', img)
cv2.waitKey(0)
img = cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=1)
cv2.imshow('after contours', img)
cv2.waitKey(0)

for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)

    if w < 5 or h < 5:
        continue

    new = img[y:y + h, x:x + w]
    new = cv2.resize(new, (64, 64))

    # cv2.imshow('images', new)
    # cv2.waitKey(0)

    cv2.imwrite('img' + str(i) + '.jpg', new)

cv2.destroyAllWindows()
