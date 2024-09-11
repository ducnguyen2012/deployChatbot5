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

def ChatBot(pathToPDF: str, query: str):
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
    )
    
    # Set up the conversational agent
    tools = [
        Tool(
            name="Personalized QA Chat System",
            func=qa.run,
            description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
        )
    ]
    
    prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                You have access to a single tool:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""
    
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    llm_chain = LLMChain(
        llm=ChatOpenAI(
            temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
        ),
        prompt=prompt,
    )
    
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    # Allow the user to enter a query and generate a response
    

    # Run the agent chain
    res = agent_chain.run({"input": query, "chat_history": []})
    
    # Check if the result is a string and convert to a dictionary
    

    return res
   

    
