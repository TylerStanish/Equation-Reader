import cv2
import math
from Operand import Operand
from Equation import Equation

# Function declarations
def get_coords_of_adjacent(ncp, ind):
    x0 = y0 = h0 = w0 = 0
    x1 = y1 = h1 = w1 = 0
    x, y, w, h = cv2.boundingRect(ncp[ind])
    if ind == 0:
        x1, y1, w1, h1, = cv2.boundingRect(ncp[ind + 1])
    elif ind == len(ncp) - 1:
        x0, y0, w0, h0, = cv2.boundingRect(ncp[ind - 1])
    else:
        x0, y0, w0, h0, = cv2.boundingRect(ncp[ind - 1])
        x1, y1, w1, h1, = cv2.boundingRect(ncp[ind + 1])
    return x0, y0, h0, w0, x, y, w, h, x1, y1, w1, h1


def check_for_arrows(image, equation, ncp, ind):
    x0, y0, h0, w0, x, y, w, h, x1, y1, w1, h1 = get_coords_of_adjacent(ncp, ind)

    if math.fabs(x - x0) < 0.4 * w and math.fabs((x0 + w0) - (x + w)) < 0.4 * w and y0 < y:
        obj = Operand(
            x=x,
            y=y,
            top=Operand(x=x0, y=y0, image=image[y0:y0 + h0, x0:x0 + w0]),
            image=image[y:y + h, x:x + w]
        )
        equation.add_operand(obj)
        return True
    elif math.fabs(x - x1) < 0.4 * w and math.fabs((x1 + w1) - (x + w)) < 0.4 * w and y1 < y:
        obj = Operand(
            x=x,
            y=y,
            top=Operand(x=x1, y=y1, image=image[y1:y1 + h0, x1:x1 + w1]),
            image=image[y:y + h, x:x + w]
        )
        equation.add_operand(obj)
        return True
    else:
        equation.add_operand(Operand(x=x, y=y, image=image[y:y + h, x:x + w]))
        return False


def read_image():
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
        if h < 4 or w < 4:
            continue
        new_contours.append((x, c))
    new_contours = sorted(new_contours, key=lambda x: x[0])
    nc = [a[1] for a in new_contours]

    equation = Equation(image=image)
    i = 0
    for c in nc:
        if i >= len(nc):
            break
        b = False
        b = check_for_arrows(image, equation, nc, i)
        # Where bool is the variable telling us whether or not to pass over the next contour
        if b:
            i += 1
        i += 1

    equation.display_equation()

