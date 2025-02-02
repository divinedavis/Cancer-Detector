# Cancer Detection Web Application (Convelutional Neural Network - Machine Learning - Deep Learning - Artificial Intelligence)

This project is a full-stack application for detecting cancer medical images. It includes a React frontend and a Flask backend to handle image uploads and predictions.
Phase 1 (MVP) - be able to detect MRI images at a 90% rate
Phase 2 - be able to detect all medical images at a 90% rate
Phase 3 - be able to detect all medical images at a 98% rate

## Features

- **Drag-and-drop interface** for image uploads.
- Sends the uploaded image to a Flask API for prediction.
- Displays prediction results to the user.

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

### **Image Processing and Prediction File**
The prediction process is handled by the **`backend.py`** file. When an image is uploaded through the `/api/predict` endpoint:

1. The image is temporarily stored on the server.
2. A deep learning model, implemented in TensorFlow/Keras, processes the image.
3. The model predicts whether the image shows signs of a brain tumor.
4. The result is returned as a JSON response to the frontend.

### **Python Libraries:**
- **Flask:** Handles API requests for image uploads and predictions.
- **Flask-CORS:** Enables cross-origin requests from the frontend.
- **TensorFlow/Keras:** Implements the deep learning model.
- **Werkzeug:** Manages file uploads securely.
- **Pandas (optional):** For handling data operations (if needed).

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

