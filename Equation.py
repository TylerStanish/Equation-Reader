import cv2


class Equation:
    operands = []
    image = ''

    def __init__(self, image):
        self.image = image

    def add_operand(self, operand):
        self.operands.append(operand)

    def add_image(self, image):
        self.image = image

    def display_equation(self):
        cv2.imshow('full equation', self.image)
        cv2.waitKey(0)
        for operand in self.operands:
            cv2.imshow('operands', operand.image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
