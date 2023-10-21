from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
app = Flask(__name__)

@app.route('/', methods = ["get", "post"])
def predict():
    message = ""
    if request.method == "POST":
        age = request.form.get('age')
        pclass = request.form.get('pclass')
        print(age, pclass)
        person = [[float(pclass), float(age), 1., 0., 0., 8.]]
        model_loaded = tf.keras.models.load_model("venv/titanic_mlp")
        pred = model_loaded.predict(person)
        print(pred)
        message = f"Survived with probability {pred[0][0]}"
    return render_template('index.html', message = message)

#model_loaded.predict([[3., 1., 29., 0., 0., 8.]])
@app.route('/text/')
def print_text():
    return "<h1>Some text</h1>"

app.run()