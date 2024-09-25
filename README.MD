# PDF_BASED_CHATBOT
A chatbot application built using Streamlit, LangChain, and OpenAI services that enables users to upload PDF files, process their contents, and ask questions about the uploaded document.

## Features
* Upload PDF files via the Streamlit interface.
* Extract and process text from the PDF.
* Split text into chunks to handle large documents efficiently.
* Use OpenAI embeddings to create vector representations of document chunks.
* Perform similarity search on document text based on user questions.
* Generate answers to user questions using OpenAI's models.
## Prerequisites
To run this project, ensure you have the following installed:

* Python 3.9+
* Streamlit
* LangChain
* PyPDF2
* FAISS 
* dotenv
* OpenAI credentials (API Key and Endpoint)

## Usage
1.**Run the Streamlit app:**
streamlit run app.py

2.**Upload a PDF file:**
Once the app is running, upload a PDF file via the sidebar.

3.**Ask a question:**
After uploading, type your question in the input box to ask the chatbot.

4.**Receive answers:** 
The chatbot will search for relevant sections of the PDF and generate an answer using the Azure OpenAI API.