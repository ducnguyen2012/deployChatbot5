import os 
import json
import pandas as pd
import traceback
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
# from langchain .llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
# import PyPDF2
#! remember to unplug api key before git push!
import re
# import time
from io import BytesIO
from typing import List




from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from apiKey import api

def parse_pdf(file: BytesIO) -> List[str]:
    '''
    preprocessing file pdf.
    input: pdf file path
    
    return: list of string
    '''
    pdf = PdfReader(file) #! read content from pdf
    output = []
    #print(pdf.pages) # pdf.pages will result a list of pages type
    for page in pdf.pages:
        text = page.extract_text() #! get text in each page
        # Merge word which contant dash in the middle. Ex: a-b
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output
def GenerateMCQ(NUMBER: int, SUBJECT: str, TONE: str):
    
    load_dotenv() #take environment variables from .env 
    # First writedown your own OPENAI API KEY
    KEY = os.getenv(api())
    llm = ChatOpenAI(openai_api_key=KEY, model_name='gpt-3.5-turbo', temperature='0.5')
    RESPONSE_JSON = {
        "1": {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
        "2": {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
        "3": {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
    }
    TEMPLATE_GENERATION="""
    Text{text}
    You are an expert multiple choice questions maker. Given the above text, it is your job to \
    create a quiz of {number} multiple choice questions for {subject} primary students in {tone} tone.
    Make sure the questions are not repeated and check all the questions to be confirmed the text as well.
    Make sure the language should be simple, clear and engaging for young learners. \
    Make sure the question should be faithfullness compare to data provide. \
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make {number} MCQs
    ### RESPONSE_JSON
    {response_json}

    """
    TEMPLATE_EVALUATION="""
    You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} primary students.\
    You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 100 words for complexity analysis. \
    if the quiz is not at per with the cognitive and analytical abilities of the students,\
    update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities. \
    Quiz_MCQs:
    {quiz}

    Check from an expert English Writer of the above quiz:
    """
    quize_evaluation_prompt = PromptTemplate(
        input_variables=['subject','quiz'], 
        template=TEMPLATE_EVALUATION
    )
    quize_generation_prompt = PromptTemplate(
        input_variables=['text','number','subject','tone','response_json'],
        template=TEMPLATE_GENERATION
    )
    quiz_chain = LLMChain(llm=llm, prompt=quize_generation_prompt, output_key='quiz', verbose=True)
    review_chain = LLMChain(llm=llm, prompt=quize_evaluation_prompt, output_key='review', verbose=True)
    generate_evaluate_chain = SequentialChain(
        chains=[quiz_chain, review_chain], 
        input_variables=['text','number','subject','tone','response_json'],
        output_variables=['quiz','review'],
        verbose=True
    )

    

    
    pdf_file_path = '../../pdfData/Cells and Chemistry of Life.pdf'
    
    pdf_text = parse_pdf(pdf_file_path)
    text = ''.join(i for i in pdf_text)
    # Conveting the python dictionary into a JSON-formatted string
    json.dumps(RESPONSE_JSON)
    
    
    # get_openai_callback() is used to set up token usage tracking in langchain
    with get_openai_callback() as cb:
        response = generate_evaluate_chain(
            {
                'text':text[:40000],
                'number':NUMBER,
                'subject':SUBJECT,
                'tone':TONE,
                'response_json': json.dumps(RESPONSE_JSON)
            }
        )

    print(f"Total Tokens:{cb.total_tokens}")
    print(f"Prompt Tokens:{cb.prompt_tokens}")
    print(f"Completion Tokens:{cb.completion_tokens}")
    print(f"Total Cost:{cb.total_cost}")
    quiztemp = response['quiz']
    quiztemp = quiztemp.replace("### RESPONSE_JSON","")
    quiz = json.loads(quiztemp)
    return quiz


