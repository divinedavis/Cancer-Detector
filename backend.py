# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Directory to save uploaded images
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dummy prediction function
def predict_image(file_path):
    # Replace this with actual model logic
    return "No Tumor Detected"

@app.route('/api/predict', methods=['POST'])
def predict():
    # Ensure the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    # Run prediction
    prediction = predict_image(file_path)

    # Return the prediction result
    return jsonify({'prediction': prediction})

# Health check endpoint (optional)
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server is running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
