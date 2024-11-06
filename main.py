import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Upload PDF files
st.header("My Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text from PDF
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generating embeddings
    embeddings = AzureOpenAIEmbeddings(openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                       azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                       model="text-embedding-ada-002",
                                       api_version=os.environ["AZURE_OPENAI_API_VERSION"],)

    # Creating vector store using FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user question
    user_question = st.text_input("Type your question here")

    if user_question:
        #define the LLM
        llm = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                              azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                              openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],)


        # Set up RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Or use "map_reduce" or "refine" depending on your need
            retriever=vector_store.as_retriever()
        )

        # Run the chain and get the response
        response = qa_chain.run(user_question)
        st.write(response)




