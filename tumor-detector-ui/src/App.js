import React from 'react';
import DragDropImageUpload from './components/DragDropImageUpload';

function App() {
    console.log('App.js component rendered!');  // Add this log

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
