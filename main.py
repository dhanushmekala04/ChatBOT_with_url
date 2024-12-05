# Import necessary libraries
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Constants
api_key = "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"
DB_DIR = "C:\\Users\\HP\\OneDrive\\Desktop\\New folder (3)\\db"

# Function to get Groq LLM
def get_groq_llm():
    model = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")
    return model

# Function to get response from Groq LLM
def get_response(user_input, retriever):
    llm = get_groq_llm()
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': user_input})
    return response['answer']

# Function to create vector store from URL
def get_vectorstore_from_url(url, db_dir=DB_DIR):
    # Load data from the specified URL
    loader = WebBaseLoader(url)
    data = loader.load()
    
    # Split the loaded data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_document_chunks = text_splitter.split_documents(data)

    # Create embeddings using Hugging Face SentenceTransformer
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})

    # Create a FAISS vector database from the documents
    vector_store = FAISS.from_documents(final_document_chunks, embeddings)
    
    return vector_store.as_retriever()

# Streamlit app configuration
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    db_dir = st.text_input("Database Directory", value=DB_DIR)

if website_url and db_dir:
    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "retriever" not in st.session_state:
        st.session_state.retriever = get_vectorstore_from_url(website_url, db_dir)

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query, st.session_state.retriever)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
