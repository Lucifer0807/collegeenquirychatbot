import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("LNMIIT Enquiry Chatbot")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

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

# Function to create the vector embeddings
def vector_embedding(max_retries=3):
    for attempt in range(max_retries):
        try:
            if "vectors" not in st.session_state:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.loader = PyPDFDirectoryLoader("./CollegePDf")  # Data Ingestion
                st.session_state.docs = st.session_state.loader.load()  # Document Loading
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings
                st.write("Vector Store DB Is Ready.")
                break  # Exit the loop if successful
        except Exception as e:
            st.write(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(5)  # Wait before retrying
    else:
        st.write("Failed to create the vector store DB after multiple attempts. Please try again later.")

prompt1 = st.text_input("Enter Your Question")

if st.button("Documents Embedding"):
    vector_embedding()

if prompt1:
    try:
        if "vectors" not in st.session_state:
            st.write("Please create the vector store DB first by clicking the 'Documents Embedding' button.")
        else:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time:", time.process_time() - start)
            st.write(response['answer'])

            # With a Streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
    except Exception as e:
        st.write(f"An error occurred while processing your request: {str(e)}")
