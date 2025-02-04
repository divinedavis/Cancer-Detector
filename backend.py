from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Directory to save uploaded images
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Correct path to your model file
model_path = 'C:/Users/divin/Documents/Cancer Detector/models/final_brain_tumor_model.h5'

# Load the trained model
try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Class labels
class_labels = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}

def predict_image(file_path):
    # Load and preprocess the image
    image = load_img(file_path, target_size=(150, 150), color_mode='grayscale')
    image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Get prediction from the model
    prediction = model.predict(image)
    print("Raw prediction output:", prediction)

    # Handle both binary and multi-class classification outputs
    if prediction.shape[-1] > 1:
        predicted_class = np.argmax(prediction)  # Multi-class case
    else:
        predicted_class = int(prediction > 0.5)  # Binary classification case

    print("Predicted class:", predicted_class)
    result = class_labels.get(predicted_class, 'Unknown')

    return result

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    # Run prediction
    prediction = predict_image(file_path)

    return jsonify({'prediction': prediction})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
