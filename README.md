# Caner Detection Application

This project is a full-stack application for detecting cancer in medical images. It includes a React frontend, Flask backend to handle image uploads and predictions, and a Neural Network to predict outcomes.

## Features

- **Drag-and-drop interface** for image uploads.
- Sends the uploaded image to a Flask API for prediction.
- Displays prediction results to the user.

## Feature Implementation
This project is a full-stack application for detecting cancer medical images. It includes a React frontend and a Flask backend to handle image uploads and predictions.

- Phase 1 (MVP) - be able to detect MRI images at a 90% rate
- Phase 2 - be able to detect all medical images at a 90% rate
- Phase 3 - be able to detect all medical images at a 98% rate

---

## Project Structure

```
project-root/
    src/
        App.js                 # Main React component
        index.js               # Entry point for React app
        components/
            DragDropImageUpload.js  # Component for drag-and-drop uploads
        styles/
            DragDropImageUpload.css  # Styling for the UI
    public/
        index.html             # HTML template for React
    backend.py                 # Flask backend API
    package.json               # React app dependencies and scripts
```

---

## Prerequisites

Make sure you have the following installed:

- **Node.js** (for the React frontend)
- **Python 3.x** (for the Flask backend)

Install dependencies:

### For the Frontend:
```sh
cd project-root
npm install
```

### For the Backend:
```sh
pip install -r requirements.txt
```

*Note:* You may need to install Flask and Flask-CORS if not listed in `requirements.txt`:
```sh
pip install flask flask-cors
```

---

## How to Run

### Start the Backend:
```sh
python backend.py
```
- The backend runs on **http://127.0.0.1:5000**.

### Start the Frontend:
```sh
npm start
```
- The React app runs on **http://localhost:3000**.

---

## Testing the App

1. Open **http://localhost:3000** in your browser.
2. Drag and drop an image into the provided drop area.
3. Click **Submit Image**.
4. View the prediction result displayed below the button.

---

## API Endpoints

### `/api/predict` (POST)
- **Description:** Accepts an image file and returns a prediction result.
- **Request:**
    - Form data with a key `file` containing the image.
- **Response:**
```json
{
  "prediction": "No Tumor Detected"
}
```

### `/api/health` (GET)
- **Description:** Health check endpoint.
- **Response:**
```json
{
  "status": "Server is running"
}
```

---

## Technologies Used

- **Frontend:** React
- **Backend:** Flask
- **Styling:** CSS (Drag-and-drop interface)
- **Deep Learning:** TensorFlow, Keras, and Convolutional Neural Networks (CNN)

### **Deep Learning Architecture:**
The backend model uses TensorFlow and Keras to implement a CNN designed for classifying brain tumor images. Key components of the architecture include:

- **Convolution Layers:** Extract spatial features from input images.
- **Max Pooling Layers:** Reduce the dimensionality of feature maps while retaining essential information.
- **Flatten and Dense Layers:** Fully connected layers that process extracted features for final classification.
- **Dropout Layers:** Prevent overfitting by randomly deactivating nodes during training.

This architecture is optimized for handling medical images and performing binary or multi-class predictions.

---

## Overview of Your Network Architecture
This architecture is defined in the `build_model` function in `cancer_trainer.py`:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
])
```

### **Components Explained:**

1. **Convolution Layers (`Conv2D`)**
   - Extract spatial features from input images by applying filters.
   - The filters learn patterns such as edges and shapes.

2. **Max Pooling Layers (`MaxPooling2D`)**
   - Downsample (reduce) the size of feature maps, keeping essential information while lowering computation.

3. **Flatten Layer (`Flatten`)**
   - Converts the 2D feature maps into a 1D vector to pass to fully connected layers.

4. **Dense Layers (`Dense`)**
   - Fully connected layers that perform classification based on the extracted features.

5. **Dropout Layer (`Dropout`)**
   - Randomly deactivates neurons during training to reduce overfitting and improve generalization.

6. **Output Layer**
   - Uses **softmax** for multi-class classification (e.g., brain tumor types) or **sigmoid** for binary classification (e.g., tumor vs. no tumor).

### **Key Characteristics of Your CNN:**

- **Small to medium complexity:** Suitable for image sizes around 150x150 or 299x299.
- **Classification focus:** Detects categories such as **glioma**, **meningioma**, **no_tumor**, and **pituitary** tumors.

---

## Troubleshooting

### Common Issues:

1. **Frontend shows default template:**
   - Ensure that React is serving updated files.
   - Clear the browser cache and hard reload the page.

2. **CORS issues:**
   - Make sure Flask-CORS is enabled in `backend.py`:
   ```python
   from flask_cors import CORS
   app = Flask(__name__)
   CORS(app)
   ```

3. **WebSocket pending:**
   - Disable fast refresh in `package.json`:
   ```json
   "scripts": {
     "start": "FAST_REFRESH=false react-scripts start"
   }
   ```
   - Restart the app with `npm start`.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

Feel free to fork and contribute to the project!

