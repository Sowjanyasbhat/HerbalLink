from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
import numpy as np
import os
import datetime

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model('leaf_model.h5')

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.herballink
collection = db.scans

# ✅ Corrected list of actual leaf names
class_names = [
    'Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala',
    'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja',
    'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly',
    'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)',
    'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus',
    'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine',
    'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass',
    'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem',
    'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)',
    'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin',
    'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala',
    'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi',
    'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala'
]

# Example medicinal uses
leaf_info = {
    "Neem": "Used for skin infections, purifies blood.",
    "Tulsi": "Boosts immunity, treats cold and fever.",
    "Guava": "Supports digestion and blood sugar control.",
    "Mango": "Aids digestion and builds immunity.",
    "Aloevera": "Soothes skin and helps digestion."
}

# Skin-related medicinal uses dictionary
skin_uses_dict = {
    "Neem": "Used for eczema, acne, and other skin infections.",
    "Aloevera": "Soothes burns, reduces scars, and hydrates skin.",
    "Tulsi": "Helps with acne, fungal infections, and rashes."
}

@app.route('/')
def home():
    return render_template('scan.html')

@app.route('/predict-leaf', methods=['POST'])
def predict_leaf():
    file = request.files['image']  # Get uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Prepare image for prediction
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict leaf
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    # ✅ Threshold to detect "Not a Leaf"
    CONFIDENCE_THRESHOLD = 20.0  # You can adjust this

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_leaf = "Not a Leaf"
        uses = "No medicinal information available."
    else:
        predicted_leaf = class_names[predicted_index]
        uses = leaf_info.get(predicted_leaf, "No info available")

    # Save result to MongoDB
    collection.insert_one({
        "leaf_name": predicted_leaf,
        "uses": uses,
        "confidence": f"{confidence:.2f}%",
        "filename": file.filename,
        "timestamp": datetime.datetime.now()
    })

    return jsonify({
        "leaf_name": predicted_leaf,
        "uses": uses,
        "confidence": f"{confidence:.2f}%"
    })

# ✅ Route for leaf info page
@app.route('/leaf-info')
def leaf_info_page():
    return render_template('leaf_info.html')

# ✅ API to get skin-related info for a leaf
@app.route('/get-leaf-info')
def get_leaf_info():
    leaf_name = request.args.get('leaf')
    skin_uses = skin_uses_dict.get(leaf_name, "No specific skin disease use found.")
    return jsonify({"skin_uses": skin_uses})

if __name__ == '__main__':
    app.run(debug=True)
