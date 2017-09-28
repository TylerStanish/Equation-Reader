import cv2
import numpy as np

# our function declarations
def get_contour_areas(contours):
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    return areas

def x_coord_contour(contour):
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
    return

def sort_left_right(contours):
    return sorted(contours, key=x_coord_contour, reverse=False)

########## Ok here's where it really starts

image = cv2.imread('IMG_1330.JPG')
cv2.imshow('direct', image)
# image = cv2.resize(image, (700, 500))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.fastNlMeansDenoising(gray)
cv2.imshow('de-noised', gray)
cv2.waitKey(0)
edged = cv2.Canny(gray, 30, 130)
# edged = cv2.Canny(gray, 30, 200)
cv2.imshow('Canny edges', edged)
cv2.waitKey(0)

heirarchy, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Canny after contouring', edged)
cv2.waitKey(0)

# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
# cv2.imshow('Contours', image)
# cv2.waitKey(0)

newArr = []
for (i, c) in enumerate(sort_left_right(contours)):
    x, y, w, h = cv2.boundingRect(c)
    if h < 5 or w < 5:
        continue

    cropped = image[y:y+h, x:x+w]
    filename = 'cropped' + str(i+5) + '.jpg'
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(filename, cropped)

    newArr.append((x, cropped))

newArr = sorted(newArr, key=lambda x:x[0])
for img in newArr:
    cv2.imshow('cropped', img[1])
    cv2.waitKey(0)

cv2.destroyAllWindows()




