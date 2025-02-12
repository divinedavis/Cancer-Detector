import React, { useState } from 'react';
import '../styles/DragDropImageUpload.css';

export default function DragDropImageUpload() {
    const [image, setImage] = useState(null);
    const [message, setMessage] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loggedIn, setLoggedIn] = useState(false);

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
            setMessage('Please upload or select an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', image);

        try {
            const response = await fetch('http://127.0.0.1:5000/api/predict', {
                method: 'POST',
                body: formData,
                credentials: 'include',
            });

            if (response.ok) {
                const result = await response.json();
                setMessage(`Prediction result: ${result.prediction}`);
            } else {
                setMessage('Error occurred while processing the image.');
            }
        } catch (error) {
            setMessage(`Error: ${error.message}`);
        }
    };

    const handleSignup = async () => {
        const response = await fetch('http://127.0.0.1:5000/api/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        const result = await response.json();
        setMessage(result.message || result.error);
    };

    const handleLogin = async () => {
        const response = await fetch('http://127.0.0.1:5000/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
            credentials: 'include',
        });

        const result = await response.json();
        if (response.ok) {
            setLoggedIn(true);
        }
        setMessage(result.message || result.error);
    };

    const checkProfile = async () => {
        const response = await fetch('http://127.0.0.1:5000/api/profile', {
            method: 'GET',
            credentials: 'include',
        });

        const result = await response.json();
        setMessage(result.message || result.error);
    };

    return (
        <div className="drag-drop-container">
            <h1>Brain Tumor Image Detection</h1>

            {/* Signup Form */}
            <div className="auth-forms">
                <h2>Signup</h2>
                <input
                    type="text"
                    placeholder="Username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                />
                <input
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                />
                <button onClick={handleSignup}>Sign Up</button>
            </div>

            {/* Login Form */}
            <div className="auth-forms">
                <h2>Login</h2>
                <button onClick={handleLogin}>Log In</button>
            </div>

            {/* Protected Route Check */}
            <div className="auth-forms">
                <h2>Protected Route</h2>
                <button onClick={checkProfile}>Check Profile</button>
            </div>

            {/* Drag-and-drop area */}
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

            {/* File input for device upload */}
            <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="file-input"
            />

            <button onClick={handleSubmit} className="submit-button">
                Submit Image
            </button>

            {message && <p className="message">{message}</p>}
        </div>
    );
}
