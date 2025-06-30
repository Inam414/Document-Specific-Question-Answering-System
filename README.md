# Document-Specific-Question-Answering-System

A lightweight question-answering system that lets you upload documents (PDFs) and get answers to your questions using semantic search and a local LLM.

## Features 
📄 PDF Processing: Extracts text from uploaded PDF documents
✂️ Smart Chunking: Splits documents into meaningful sections with overlap
🔍 Semantic Search: Finds relevant document sections using FAISS vector store
💬 LLM Answers: Generates answers using Hugging Face's GPT-2 (or other models)
🖥️ Interactive Interface: Chat-style Q&A interface in the terminal/notebook

## Installation 
!pip install -q langchain-community llama-index sentence-transformers faiss-cpu pypdf chromadb transformers accelerate bitsandbytes jedi>=0.16

## Usage
Run the script
Upload your PDF when prompted
Ask questions about the document
Type 'quit' to exit
