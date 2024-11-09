from flask import Flask, render_template, request

from processing import process, preprocess


app = Flask(__name__)


@app.route('/', methods=["get", "post"])
def index():
    message = ''
    if request.method == "POST":
        coef = request.form.get("coef")
        try:
            coef = float(coef)
        except:
            coef = 0
            message += 'Некорректный ввод. Установлено значение по умолчанию. '
        scaled_coef=preprocess(coef)
        
        mu = process(scaled_coef)
        message += f"Модуль упругости при растяжении {mu} ГПа"
    return render_template("index.html", message=message)


if __name__ == "__main__":
    app.run()
