from flask import Flask
app = Flask(__name__)


@app.route('', methods=['GET', 'POST'])
def classify_image():
    return 'Now classifying image'

if __name__ == '__main__':
    app.run()

