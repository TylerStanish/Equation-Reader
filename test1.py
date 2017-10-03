import cv2
import numpy as np

img = cv2.imread('test_images/img_1330.jpg', 0)
img = cv2.fastNlMeansDenoising(img)
edged = cv2.Canny(img, 30, 130)
hierarchy, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

white = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
white[:] = (255, 255, 255)
white = cv2.drawContours(white, contours, -1, (0, 0, 0))
white = cv2.fillPoly(white, pts=contours, color=(0, 0, 0))
cv2.imshow('white', white)
cv2.waitKey(0)


for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    if h < 5 or w < 5:
        continue
    temp = white[y:y + h, x:x + w]
    cv2.imwrite('img_' + str(i+100) + '.jpg', temp)

cv2.destroyAllWindows()
