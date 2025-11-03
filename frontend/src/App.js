import React, { useState } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const handleAsk = async () => {
    if (!question) return;
    try {
      const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: question }),
      });
      const data = await response.json();
      setAnswer(data.answer);
    } catch (error) {
      setAnswer('Error connecting to backend');
    }
  };

  return (
    <div className="App" style={{ padding: 20 }}>
      <h1>Math Agent</h1>
      <input
        type="text"
        placeholder="Ask a math question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: '80%', padding: 8, marginRight: 10 }}
      />
      <button onClick={handleAsk} style={{ padding: 8 }}>
        Ask
      </button>
      <div style={{ marginTop: 20 }}>
        <strong>Answer:</strong>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default App;
