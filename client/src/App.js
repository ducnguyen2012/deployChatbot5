import logo from './logo.svg';
import './App.css';
import Home from './page/Home'

import { Route, BrowserRouter, Routes } from 'react-router-dom';
import MCQPage from './page/MCQPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/ChatBotAIWebsite' element={<Home />} />
        
        <Route path='/question' element={<MCQPage/>} />
      </Routes>
    </BrowserRouter>
    
  );
}

export default App;
