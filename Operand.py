# from main import classifier

class Operand:
    def __init__(self, x=0, y=0, top='', image='', superscript=''):
        self.x = x
        self.y = y
        self.top = top
        self.image = image
        self.superscript = superscript
        # self.latex = latex
        # self.top = top
        # self.superscript = superscript
        # self.subscript = subscript
        # self.children = children
        # self.image = image

    # def __init__(self):
    #     print('instantiated')

    @staticmethod
    def get_latex(str):
        if str == 'alpha':
            return '\alpha'
        if str == 'beta':
            return '\beta'
        if str == 'gamma':
            return '\gamma'

        return str

    # def classify(self):
    #     classifier.predict(self.image)