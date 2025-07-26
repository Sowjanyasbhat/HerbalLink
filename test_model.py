from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = load_model("leaf_model.h5")

# Class mapping (optional)
class_names = ['Aloe Vera', 'Basil', 'Curry', 'Guava', 'Lemon', 'Mango', 'Amla', 'Tulsi']

# Load and preprocess a test image
img_path = "D:\HerbalLink project\dataset\Indian Medicinal Leaves Image Datasets\Medicinal Leaf dataset\Amla\404.jpg"  # replace with actual image path
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
predicted_index = np.argmax(pred)
predicted_class = class_names[predicted_index]

print("✅ Predicted Leaf Type:", predicted_class)
