import React, { useState } from 'react';

export default function DragDropImageUpload() {
    const [image, setImage] = useState(null);
    const [message, setMessage] = useState('');

    const handleDragOver = (event) => {
        event.preventDefault();
    };

    const handleDrop = (event) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            setImage(file);
            setMessage('');
        } else {
            setMessage('Please drop a valid image file.');
        }
    };

    const handleSubmit = async () => {
        if (!image) {
            setMessage('Please upload an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', image);  // Ensure 'file' key matches backend

        try {
            setMessage('Uploading image...');

            // Fetch the backend API endpoint
            const response = await fetch('http://127.0.0.1:5000/api/predict', {
                method: 'POST',
                body: formData,
            });

            // Handle response
            if (response.ok) {
                const result = await response.json();
                setMessage(`Prediction result: ${result.prediction}`);
            } else {
                const errorResponse = await response.json();
                setMessage(`Error: ${errorResponse.error || 'Something went wrong.'}`);
            }
        } catch (error) {
            setMessage(`Error: Could not reach the server (${error.message})`);
        }
    };

    return (
        <div className="drag-drop-container">
            <h1>Brain Tumor Image Detection</h1>
            <div
                className="drop-area"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
            >
                {image ? (
                    <p>Image Selected: {image.name}</p>
                ) : (
                    <p>Drag and drop an image here</p>
                )}
            </div>
            <button onClick={handleSubmit} className="submit-button">
                Submit Image
            </button>
            {message && <p className="message">{message}</p>}
        </div>
    );
}
