import React, { useState } from 'react';
import '../styles/DragDropImageUpload.css';

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

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            setImage(file);
            setMessage('');
        } else {
            setMessage('Please select a valid image file.');
        }
    };

    const handleSubmit = async () => {
        if (!image) {
            setMessage('Please upload an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', image);

        console.log('Submitting form with image:', image.name);

        try {
            const response = await fetch('http://127.0.0.1:5000/api/predict', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Server response:', result);
                setMessage(`Prediction result: ${result.prediction}`);
            } else {
                console.log('Error from server:', response.statusText);
                setMessage('Error occurred while processing the image.');
            }
        } catch (error) {
            console.error('Error during fetch:', error);
            setMessage(`Error: ${error.message}`);
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
                {image ? <p>Image Selected: {image.name}</p> : <p>Drag and drop an image here</p>}
            </div>
            <button onClick={handleSubmit} className="submit-button">Submit Image</button>
            <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="upload-button"
            />
            {message && <p className="message">{message}</p>}
        </div>
    );
}
