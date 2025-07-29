from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model and threshold
model = load_model("covid_model.h5")
with open("best_threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if 'file' not in request.files or request.files["file"].filename == "":
            return "No file uploaded", 400

        file = request.files["file"]
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        try:
            # Preprocess
            img_loaded = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img_loaded) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            proba = model.predict(img_array)[0][0]
            result = "COVID" if proba >= threshold else "Normal"
            return f"Prediction: {result} (Confidence: {proba:.4f})"
        
        except Exception as e:
            return f"Error processing the image: {str(e)}", 500

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
