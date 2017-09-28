class operand:

    children = []
    latex = ''
    superscript = ''
    subscript = ''

    @staticmethod
    def getLatex(str):
        if str == 'alpha':
            return '\alpha'
        if str == 'beta':
            return '\beta'
        if str == 'gamma':
            return '\gamma'

        return str


