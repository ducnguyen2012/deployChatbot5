from flask import Flask, jsonify, request
from flask_cors import CORS
from PMC.ChatBot import ChatBot
from MCQModule.GenerateMCQMain import GenerateMCQ

app = Flask(__name__)
CORS(app) #! this CORS will allow request from react frontend. If it dont then will cause network error!


FILEPATH = '../../pdfData/Cells and Chemistry of Life.pdf'


@app.route('/', methods=['POST', 'GET'])
def chatbotMessage():
    try:
        userMessage = request.get_json().get('message')
        print("This is user message: " + userMessage)
        if not userMessage:
            return jsonify({"error": "No message provided!"}), 400
        
        bot_response = ChatBot(FILEPATH, userMessage)
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/question', methods=['POST'])
def uploadQuestion():
    numQuestion = request.get_json().get('numQuestions')
    subject = request.get_json().get('subject')
    tone = request.get_json().get('tone')
    print("This is num question: ", numQuestion)
    print("This is my subject: ", subject)
    print("This is my tone: ", tone)
    question = GenerateMCQ(numQuestion, subject, tone)
    return jsonify(question)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000,debug=True)
