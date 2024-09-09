import React, { useState } from 'react';
import '../CSS/MCQPage.css'; // Add necessary styles

const MCQPage = () => {
  const [questions, setQuestions] = useState([]); // set question
  const [error, setError] = useState(null); // set error
  const [numQuestions, setNumQuestions] = useState(''); // set number of question to generate
  const [subject, setSubject] = useState(''); // set subject for chatbot
  const [tone, setTone] = useState(''); // set tone for chatbot
  const [selectedOptions, setSelectedOptions] = useState({}); // set select option
  const [feedbacks, setFeedbacks] = useState({}); // set feedback after each option
  const [colors, setColors] = useState({}); // set color for text
  const [isLoading, setIsLoading] = useState(false); // manage loading

  const handleResponse = (data) => {
    const formattedQuestions = Object.values(data).map(q => ({
      question: q.mcq,
      options: Object.entries(q.options).map(([key, value]) => ({
        key,
        value
      })),
      correct: q.correct
    }));
    setQuestions(formattedQuestions);
  };

  const handleFetchQuestion = () => {
    setIsLoading(true); // Start loading
    fetch('http://127.0.0.1:5000/question', {
      method: "POST",
      headers: {
        'Content-type': 'application/json',
      },
      body: JSON.stringify({
        numQuestions,
        subject,
        tone,
      }),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error("network response not ok");
        }
        return response.json();
      })
      .then(data => {
        handleResponse(data);
        setIsLoading(false); // Stop loading after response
      })
      .catch(error => {
        setError(error.message);
        console.log("Error fetching question: ", error);
        setIsLoading(false); // Stop loading in case of error
      });
  };
  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      console.log("enter key is press")
      event.preventDefault();
      handleFetchQuestion()
    }
  }
  const handleOptionClick = (questionIndex, optionKey, correctAnswer) => {
    setSelectedOptions(prev => ({
      ...prev,
      [questionIndex]: optionKey
    }));

    if (optionKey === correctAnswer) {
      setColors(prev => ({
        ...prev,
        [questionIndex]: 'green'
      }));

      // Reset color after 1 seconds
      setTimeout(() => {
        setColors(prev => ({
          ...prev,
          [questionIndex]: 'black'
        }));
      }, 1000);
    } else {
      setColors(prev => ({
        ...prev,
        [questionIndex]: 'red'
      }));

      setTimeout(() => {
        setColors(prev => ({
          ...prev, [questionIndex]: 'black'
        }));
      }, 1000);
    }

    setFeedbacks(prev => ({
      ...prev,
      [questionIndex]: optionKey === correctAnswer ? "Correct!" : "It is not correct!"
    }));
  };

  return (
    <form className="mcq-form" onKeyDown={handleKeyPress}>
      <div className="form-group">
        <label htmlFor="numQuestions">Number of Questions:</label>
        <input
          type="text"
          id="numQuestions"
          value={numQuestions}
          onChange={(e) => setNumQuestions(e.target.value)}
          placeholder='Enter number of questions'
          className="input-field"
        />
      </div>
      <div className="form-group">
        <label htmlFor="subject">Subject:</label>
        <input
          type="text"
          id="subject"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          placeholder='Enter subject of questions'
          className="input-field"
        />
      </div>
      <div className="form-group">
        <label htmlFor="tone">level:</label>
        <input
          type="text"
          id="tone"
          value={tone}
          onChange={(e) => setTone(e.target.value)}
          placeholder='Enter level of questions'
          className="input-field"
        />
      </div>
      <button type="button" onClick={handleFetchQuestion} className="fetch-btn">Fetch Questions</button>

     

      {isLoading ? (
        <div className="loading">
          <p>Loading questions</p>
          <div className="dot-container">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        </div>
      ) : (
        <div className="questions-container">
          {questions.map((q, index) => (
            <div key={index} className="question-block">
              <h3>{index + 1}) {q.question}</h3>
              <div className="options-container">
                {q.options.map(option => (
                  <label key={option.key} className="option-label">
                    <input
                      type="radio"
                      name={`question-${index}`}
                      value={option.key}
                      onClick={() => handleOptionClick(index, option.key, q.correct)}
                      className="option-input"
                    />
                    <span style={{ color: selectedOptions[index] === option.key ? colors[index] : 'black' }}>
                      {option.value}
                    </span>
                  </label>
                ))}
              </div>
              {selectedOptions[index] && (
                <p className="feedback">Feedback: {feedbacks[index]}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </form>
  );
};

export default MCQPage;
