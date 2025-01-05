# PDFConverse
PDFConverse- A Multi-RAG Chatbot for PDF Interaction

This project implements a web application that allows users to interact with PDF documents using a conversational AI chatbot. Built with Streamlit, Langchain, FAISS, and OpenAI (or similar models), the system provides detailed answers to user queries by retrieving relevant information from uploaded PDFs.

The core functionality includes:

Text extraction: The application extracts text from PDF documents using PyPDF2 to ensure content is available for processing.

Text segmentation: It splits the extracted text into chunks using Langchain’s RecursiveCharacterTextSplitter, improving efficiency in handling large documents.

Vector-based search: Using FAISS for creating and querying a vector store, the application enables fast and efficient retrieval of relevant chunks based on user input.

Conversational AI: The chatbot leverages language models like OpenAI GPT-3.5 (or GPT-4) for natural language processing to answer queries based on the extracted document text.

Real-time interaction: Users can upload PDFs and interact with the chatbot in real-time through an intuitive Streamlit web interface.


**Key Features:**

Text Extraction: Converts PDF content into machine-readable format.

Smart Search: Finds and ranks relevant chunks using FAISS to return answers to user queries.

Customizable Prompt: Tailored prompts to ensure detailed and accurate responses.

User-Friendly Interface: Clean and simple interface powered by Streamlit for seamless user experience.


**Technologies Used:**

Streamlit – Web framework for interactive applications.

PyPDF2 – PDF text extraction.

Langchain – Text splitting and chaining for language models.

FAISS – Efficient vector search and retrieval.

OpenAI (or similar models) – Conversational AI for question answering.

Python – Backend logic and data processing.
