from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import sqlite3
import os
from dotenv import load_dotenv  # For environment variables

# Load environment variables from a .env file
load_dotenv()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Configure session secret key from environment variable
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))  # Fallback to random key if not set

# Set secure cookie options
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Configuration from environment variables
DATABASE_PATH = os.getenv('DATABASE_PATH', './users.db')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
MODEL_PATH = os.getenv('MODEL_PATH', 'C:/Users/divin/Documents/Cancer Detector/models/final_brain_tumor_model.h5')

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

def get_db_connection():
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ----------- AUTHENTICATION ROUTES ----------- #

@app.route('/api/signup', methods=['POST'])
def signup():
    """Register a new user with username and password."""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    # Basic input validation (example: min length)
    if len(username) < 3 or len(password) < 6:
        return jsonify({'error': 'Username must be 3+ characters, password 6+'}), 400

    password_hash = generate_password_hash(password)

    try:
        conn = get_db_connection()
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        conn.close()
        return jsonify({'message': 'User created successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 409
    except Exception as e:
        return jsonify({'error': f'Signup failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Authenticate a user and start a session."""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        session['user_id'] = user['id']
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    """Log out the current user by clearing the session."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/profile', methods=['GET'])
def profile():
    """Return profile info if user is authenticated."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({'message': 'This is a protected route'}), 200

# ----------- IMAGE PREDICTION ROUTE ----------- #

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict brain tumor type from an uploaded image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Process the image
        image = load_img(file_path, target_size=(150, 150), color_mode='grayscale')
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Run prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))  # Highest probability as confidence score

        # Map prediction to label
        class_labels = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}
        result = class_labels.get(predicted_class, 'Unknown')

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

# ----------- MAIN APP STARTUP ----------- #

if __name__ == '__main__':
    # Initialize database
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.close()

    app.run(host='0.0.0.0', port=5000, debug=True)