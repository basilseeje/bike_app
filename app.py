from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hr = float(request.form["hr"])
    temp = float(request.form["temp"])
    hum = float(request.form["hum"])
    windspeed = float(request.form["windspeed"])

    final_features = np.array([[hr, temp, hum, windspeed]])
    prediction = model.predict(final_features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Bike Rentals: {int(prediction[0])}"
    )

if __name__ == "__main__":
    app.run(debug=True)
