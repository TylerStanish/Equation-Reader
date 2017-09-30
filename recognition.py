import cv2
import math
from Operand import Operand
from Equation import Equation


# Function declarations
def check_for_arrows(nc, i):
    x0 = y0 = h0 = w0 = 0
    x1 = y1 = h1 = w1 = 0
    x, y, w, h = cv2.boundingRect(nc[i])
    if i == 0:
        x1, y1, w1, h1, = cv2.boundingRect(nc[i + 1])
    elif i == len(nc) - 1:
        x0, y0, w0, h0, = cv2.boundingRect(nc[i-1])
    else:
        x0, y0, w0, h0, = cv2.boundingRect(nc[i-1])
        x1, y1, w1, h1, = cv2.boundingRect(nc[i+1])

    if math.fabs(x - x0) < 0.4 * w and math.fabs((x0 + w0) - (x + w)) < 0.4 * w and y0 < y:
        obj = Operand(
            x=x,
            y=y,
            top=Operand(x=x0, y=y0, image=image[y0:y0 + h0, x0:x0 + w0]),
            image=image[y:y + h, x:x + w]
        )
        equation.add_operand(obj)
    elif math.fabs(x - x1) < 0.4 * w and math.fabs((x1 + w1) - (x + w)) < 0.4 * w and y1 < y:
        obj = Operand(
            x=x,
            y=y,
            top=Operand(x=x1, y=y1, image=image[y1:y1 + h0, x1:x1 + w1]),
            image=image[y:y + h, x:x + w]
        )
        equation.add_operand(obj)
    else:
        equation.add_operand(Operand(x=x, y=y, image=image[y:y + h, x:x + w]))


image = cv2.imread('IMG_1330.JPG')
cv2.imshow('direct', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.fastNlMeansDenoising(gray)

edged = cv2.Canny(gray, 30, 130)

hierarchy, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Now to sort out the contours
new_contours = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if h < 5 or w < 5:
        continue
    new_contours.append((x, c))
new_contours = sorted(new_contours, key=lambda x: x[0])
nc = [a[1] for a in new_contours]

equation = Equation(image=image)

for (i, c) in enumerate(nc):
    check_for_arrows(nc, i)

equation.display_equation()
