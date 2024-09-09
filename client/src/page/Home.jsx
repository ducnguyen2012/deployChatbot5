import React, { useState } from 'react';
import '../CSS/Home.css';
import { useNavigate } from 'react-router-dom';
import chatBotIcon from '../icon/chatBotIcon.png'
import userIcon from '../icon/userIcon.png'
const Home = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const navigate = useNavigate();

  const handleSend = async () => {
    if (input.trim() === 'MCQ') {
      navigate("/question");
    }
    if (input.trim() !== '') {
      setMessages([...messages, { text: input, sender: 'user' }]);
      setIsThinking(true);
      try {
        const response = await fetch('http://127.0.0.1:5000/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: input }),
        });
  
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
  
        const data = await response.json();
        const botMessage = { text: data.response, sender: 'bot' };
        setMessages(prevMessages => [...prevMessages, botMessage]);
      } catch (error) {
        const errorMessage = { text: 'Network error: ' + error.message, sender: 'bot' };
        setMessages(prevMessages => [...prevMessages, errorMessage]);
      } finally {
        setIsThinking(false);
      }
      setInput('');
    }
  };

  // Function to handle Enter key press
  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <div>
      <div className='uploadPDF'></div>
      <div className="chat-container">
        <div className="chat-box">
          {messages.map((message, index) => (
            <div key={index} className={`chat-bubble ${message.sender}`}>
              <div className="message-content">
                <img
                  src={message.sender === 'user' ? userIcon : chatBotIcon}
                  alt={`${message.sender} icon`}
                  className="message-icon"
                />
                <span className="message-text">{message.text}</span>
              </div>
            </div>
          ))}
          {isThinking && (
            <div className='chatbot-is-thinking'>
              <p>I am thinking about your question hihi</p>
              <div className='dot'></div>
              <div className='dot'></div>
              <div className='dot'></div>
            </div>
          )}
        </div>
        <div className="input-box">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type a message..."
            onKeyPress={handleKeyPress} 
          />
          <button onClick={handleSend}>Send</button>
        </div>
      </div>
    </div>
  );
};

export default Home;
