from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
import numpy as np
import os
import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model('leaf_model.h5')

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.herballink
collection = db.scans

# Leaf name to medicinal uses (you can expand this)
leaf_info = {
    "Neem": "Used for skin infections, purifies blood.",
    "Tulsi": "Boosts immunity, treats cold and fever.",
    "Guava": "Supports digestion and blood sugar control.",
    "Mango": "Aids digestion and builds immunity.",
    "Aloe Vera": "Soothes skin and helps digestion.",
    "Basil": "Reduces inflammation and supports lungs.",
    "Curry": "Good for eyesight and anemia.",
    "Lemon": "Rich in Vitamin C and boosts immunity."
}

@app.route('/')
def home():
    return render_template('scan.html')  # Or main_page.html if you have navigation

@app.route('/predict-leaf', methods=['POST'])
def predict_leaf():
    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Prepare image for prediction
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_names = list(leaf_info.keys())  # Ensure order matches training
    predicted_index = np.argmax(prediction)
    predicted_leaf = class_names[predicted_index]
    uses = leaf_info.get(predicted_leaf, "No info available")

    # Save result to MongoDB
    collection.insert_one({
        "leaf_name": predicted_leaf,
        "uses": uses,
        "filename": file.filename,
        "timestamp": datetime.datetime.now()
    })

    return jsonify({"leaf_name": predicted_leaf, "uses": uses})

if __name__ == '__main__':
    app.run(debug=True)
