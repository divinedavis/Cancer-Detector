import React from 'react';
import DragDropImageUpload from './components/DragDropImageUpload';
import './styles/DragDropImageUpload.css';

function App() {
    console.log('App component rendered!');  // Add this line

    return (
        <div className="App">
            <header className="App-header">
                <h1>Brain Tumor Image Detection</h1>
            </header>
            <main>
                <DragDropImageUpload />
            </main>
        </div>
    );
}

export default App;
