import streamlit as st
import os
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setting up Streamlit interface
st.title("Intoxalock Knowledge Base Assistant")
st.write("Ask questions about Intoxalock and get answers based on the content.")

# Load and index website content if it’s the first run
@st.cache_resource
def load_and_index_website():
    loader = WebBaseLoader(
        web_paths=("https://knowledge.intoxalockapp.com/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Embeddings and document indexing
    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model="text-embedding-ada-002",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=embeddings
    )
    return vectorstore.as_retriever()

retriever = load_and_index_website()

# Define the question-answering chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit input/output
user_question = st.text_input("Enter your question:", "")
if st.button("Get Answer"):
    if user_question:
        response = rag_chain.invoke({"input": user_question})
        st.write("**Answer:**")
        st.write(response.get("answer", "Sorry, I couldn't retrieve an answer."))
    else:
        st.write("Please enter a question.")
