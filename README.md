# Brain Tumor Detection Web App

This project is a full-stack application for detecting brain tumors in medical images. It includes a React frontend and a Flask backend to handle image uploads and predictions.

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
- **Deep Learning Architecture:** Convolutional Neural Network (CNN)

### **Architecture Details:**
The backend uses a CNN model to classify brain tumor images. The architecture includes:

- **Convolution Layers:** Extract spatial features from images.
- **Max Pooling Layers:** Downsample feature maps to reduce dimensionality.
- **Flatten and Dense Layers:** Fully connected layers for classification.
- **Dropout Layers:** Prevent overfitting by randomly deactivating nodes during training.

This architecture is designed to efficiently process medical images and make binary or multi-class predictions.

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

