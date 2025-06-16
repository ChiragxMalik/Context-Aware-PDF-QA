# AskMyPDF - Context-Aware PDF Q&A System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and ask questions about their content.

![Screenshot (197)](https://github.com/user-attachments/assets/69ef732e-ef1e-4d1d-93e5-09ee55942f18)

![Screenshot (198)](https://github.com/user-attachments/assets/373e2b7f-9fb7-4bda-909f-8d57ef5cb5f8)

## Features
- Upload and process PDF documents
- Ask natural language questions about the content
- Get answers with source document references
- Persistent vector storage for processed documents
- Clean UI with Streamlit

## Development Journey

During development, I encountered and solved several challenges:

1. **Initial Context Issues**: The system initially went out of context frequently. I fixed this by refining the prompt template to strictly focus on the provided context.

2. **Persistent Storage**: After testing, I realized it would be better to save vector embeddings permanently rather than processing them temporarily each time.

3. **Filename Handling**: ChromaDB doesn't allow special characters in collection names. I initially tried using `strip()` but found it insufficient for files with unusual names. The solution was to implement a robust cleaning function (`get_clean_collection_name`) that properly sanitizes filenames.

4. **Performance Optimization**: Through testing, I adjusted chunk sizes and overlap to balance between context retention and processing efficiency.

## Prerequisites
- Python 3.8 or higher
- Ollama LLM model installed and running locally (see [Ollama documentation](https://ollama.ai/docs) for setup)
- All required Python libraries installed (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chiragxmalik/Context-Aware-PDF-QA.git
   cd Context-Aware-PDF-QA
