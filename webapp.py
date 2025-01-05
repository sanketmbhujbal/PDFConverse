import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Only add non-empty text
                text += page_text
    if not text:
        raise ValueError("No extractable text found in the provided PDFs.")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)

    if not chunks:  # Check if no chunks are generated
        raise ValueError("Text splitting failed. No chunks created.")
    return chunks

def get_vector_store(text_chunks):
    # Set up embeddings with custom OpenAI base URL
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE_URL")  # Custom Base URL
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Custom prompt template
    prompt_template = """
    You are an expert AI assistant tasked with answering questions based on the provided context. Your responses should be:

    1. Highly detailed and well-structured, yet concise.
    2. Provide step-by-step explanations if necessary.
    3. If the answer cannot be found in the context, state clearly: "The information is not available in the context." 

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Configure OpenAI Chat Model
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE_URL")  # Custom Base URL
    )

    # Create QA Chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    # Load embeddings and vector database
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE_URL")  # Custom Base URL
    )
    index_path = "faiss_index"

    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index not found. Please process and save the index first.")

    new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


    # Perform similarity search
    docs = new_db.similarity_search(user_question)

    if not docs:  # No similar documents found
        st.write("No relevant context found. Please refine your question.")
    

    # Process query with conversational chain
    chain = get_conversational_chain()
    response = chain(
    {"input_documents": docs, "question": user_question},
    return_only_outputs=True)

    if "output_text" not in response:
        st.write("Unable to generate a response. Please try again.")

    # Display response
    st.write("### Response: \n", response["output_text"])

#Streamlit page setup
st.set_page_config(page_title="PDFConverse",page_icon="PDFConverse.ico",layout="wide")
st.sidebar.image("PDFConverse.webp",width=100)
st.markdown('<div class="centered-container">', unsafe_allow_html=True)
st.title("PDFConverse\n\n")
st.write("### Multi-RAG Chatbot for PDF Interaction")
st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.header("PDFConverse \n\n",divider="orange")
st.sidebar.header("Upload PDF Files")
uploaded_files = st.sidebar.file_uploader("\nChoose PDF files\n", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        # Process uploaded PDFs
        raw_text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(raw_text)

        # Create vector store
        get_vector_store(text_chunks)
        st.success("PDFs Processed Successfully!")

# User query input
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about the PDFs:", placeholder="Search...")
    st.markdown('</div>', unsafe_allow_html=True)
if user_question:
    with st.spinner("Fetching response..."):
        user_input(user_question)
