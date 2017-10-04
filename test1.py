import cv2
import numpy as np

img = cv2.imread('test_images/img_8.jpg', 0)
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# Perhaps add a border here?!?!


img = cv2.imread('test_images/img_8.jpg', 0)
# img = cv2.imread('extracted_images/')
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.waitKey(0)

# canny = cv2.Canny(img, 30, 130)
_, contours, __ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(len(contours))

for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    if h < 4 or w < 4:
        continue
    # temp = img[y:y + h, x:x + w]

    # kernel = np.ones((3, 3), np.uint8)
    # temp = cv2.dilate(temp, kernel=kernel, iterations=1)
    # temp = cv2.resize(temp, (128, 128))

    white = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    white[:] = (255, 255, 255)
    white = cv2.drawContours(white, contours, -1, (0, 0, 0), thickness=1)

    white = white[y:y+h, x:x+w]

    cv2.imshow(str(i), white)
    cv2.waitKey(0)
    cv2.imwrite('img_' + str(i+100) + '.jpg', white)

cv2.destroyAllWindows()











# img = cv2.imread('test_images/img_1330.jpg', 0)
# img = cv2.resize(img, (0, 0), fx=5, fy=5)
# img = cv2.fastNlMeansDenoising(img)
# edged = cv2.Canny(img, 30, 130)
# hierarchy, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)







# white = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
# white[:] = (255, 255, 255)
# white = cv2.drawContours(white, contours, -1, (0, 0, 0), thickness=1)
# white = cv2.fillPoly(white, pts=contours, color=(0, 0, 0))
# # for c in contours:
# #   white = cv2.fillConvexPoly(white, points=c, color=(0, 0, 0))
#
# white = cv2.resize(white, (0, 0), fx=6, fy=6)
#
# kernel = np.ones((4, 4), np.uint8)
# white = cv2.dilate(white, kernel, iterations=3)
# cv2.imshow('white', white)
# cv2.waitKey(0)