import re
from io import BytesIO
from typing import List
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from langchain.prompts import PromptTemplate
from apiKey import api


#! api key: 
api = api()

#! --------------------------- preprocessing data from pdf file ------------------------------
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

def text_to_docs(text: str) -> List[Document]:
    """
    Converts a string or list of strings to a list of Documents
    with metadata.
    """
    
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i})
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

#! 
def test_embed(page):
    embedding = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    #! safe page to vector database
    index = FAISS.from_documents(page,embedding)
    return index



def generate_question(context: str) -> str:
    """
    Generates a question based on the provided context.
    """
    llm = ChatOpenAI(openai_api_key=api)

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="Based on the following context, generate a relevant question for a primary school student:\n\n{context}\n\n"
    )

    # Create the LLMChain
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    
    # Run the chain with the context
    question = chain.run({"context": context})
    
    return question


def EvalChatBot(pathToPDF: str, numberOfQuestion: int):
    uploaded_file = pathToPDF
    
    # Parse the PDF and convert text to documents
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    
    # Test the embeddings and save the index in a vector database
    index = test_embed(pages)
    
    # Set up the question-answering system
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=api),
        chain_type="map_reduce",
        retriever=index.as_retriever(),
        return_source_documents=True
    )
    
    # Generate and answer questions
    generated_questions = []
    for i in range(numberOfQuestion):
        # Retrieve a random context to generate a question from
        random_page = pages[i % len(pages)]
        context = random_page.page_content
        
        # Generate the question based on the context
        question = generate_question(context)
        
        # Use the chatbot to answer the generated question
        result = qa({"query": question})
        answer = result['result']
        source_documents = result['source_documents']
        
        contexts = [doc.page_content for doc in source_documents]
        
        generated_questions.append({
            "question": question,
            "answer": answer,
            "contexts": contexts
        })
    #! return a list of dictionary!
    return generated_questions
# dict = EvalChatBot("../../pdfData/Cells and Chemistry of Life.pdf", 5)
# print(dict)
# print("This is len of dict: " + str(len(dict)))

   

    
