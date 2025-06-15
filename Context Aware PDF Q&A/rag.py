from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re

#loading the pdf file
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

#for chunking the documents
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

#for converting chunks to embeddings
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#for cleaning collection name as Chroma doesn't allow special characters
def get_clean_collection_name(file_path):
    file_name = os.path.basename(file_path).replace(".pdf", "")
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file_name)
    clean_name = re.sub(r"^[^a-zA-Z0-9]+", "", clean_name)
    clean_name = re.sub(r"[^a-zA-Z0-9]+$", "", clean_name)
    return clean_name

#for creating and storing the vector database
def store_embeddings(chunks, embedding_model, collection_name, persist_directory="db"):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    return vector_db

# For saving the vector store to disk............. not needed now as using streamlit
if __name__ == "__main__":
    file_path = "sample.pdf"
    persist_directory = "db"
    collection_name = get_clean_collection_name(file_path)

    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    embedding_model = get_embeddings_model()
    vector_db = store_embeddings(chunks, embedding_model, collection_name, persist_directory=persist_directory)

    print("Vector DB created and saved.")