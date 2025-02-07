import React, { useState } from 'react';
import '../styles/DragDropImageUpload.css';

export default function DragDropImageUpload() {
    const [image, setImage] = useState(null);
    const [message, setMessage] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loggedIn, setLoggedIn] = useState(false);

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
            credentials: 'include', // Send cookies with the request
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
            credentials: 'include', // Send cookies with the request
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

            {message && <p className="message">{message}</p>}
        </div>
    );
}
