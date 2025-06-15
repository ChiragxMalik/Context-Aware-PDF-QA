import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from rag import load_pdf, split_documents, get_embeddings_model, store_embeddings, get_clean_collection_name

#streamlit UI setup
st.set_page_config(page_title="AskMyPDF", layout="centered")
st.title("Context Aware PDF Q&A")

#for sidebar instructions
def main():
    st.sidebar.title("📝 AskMyPDF")
    with st.sidebar:
        st.title("Instructions")
        st.markdown("""
        **How to use this RAG application:**
        
        1. Upload a PDF document
        2. Ask questions about the content
        3. Get answers based on the document
        
        **Tips:**
        - Be specific with your questions
        - The system only knows what's in your document
        - Try questions like:
          - "Summarize the document"
          - "What does this document say about X?"
          - "What is X?"
        """)
        
        st.divider()

if __name__ == "__main__":
    main()
#Prompt Tuning Template
custom_prompt = PromptTemplate.from_template("""

You are a friendly, patient, and intelligent assistant designed to help users understand and analyze PDF documents.

Your tasks:
- Answer questions using only the provided context.
- Summarize or highlight key information when asked.
- If the answer isn't in the context, politely say so and suggest rephrasing the question or uploading a relevant PDF.
- Avoid using external knowledge or guessing.

Style: Keep responses clear, concise, human, and warm. Avoid speculations, inner monologues, or over-explaining.

---
Context:
{context}
---
Question:
{question}

Your Answer:
""")


#PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    file_name = uploaded_file.name
    collection_name = get_clean_collection_name(file_name)
    base_name = os.path.splitext(file_name)[0]  # Sanitize collection name
    persist_directory = f"./db/{base_name}"
    print(f"Collection name: {collection_name}")  # Debug print (optional)

    # 4. Save PDF & create vector DB if not already exists
    if not os.path.exists(persist_directory):
        os.makedirs("uploads", exist_ok=True)
        file_path = f"./uploads/{file_name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner("Processing your PDF..."):
            documents = load_pdf(file_path)
            chunks = split_documents(documents)
            embedding_model = get_embeddings_model()
            store_embeddings(chunks, embedding_model, collection_name=collection_name, persist_directory=persist_directory)
            st.success("PDF processed and vector DB created!")
    else:
        st.info("This PDF has already been processed. Loading from saved database...")

    # 5. Load DB and LLM
    embedding_model = get_embeddings_model()
    vector_db = Chroma(
        collection_name=collection_name,  # Use sanitized collection name
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    llm = OllamaLLM(model="gemma3:4b", temperature=0.7)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    # 6. Question Input
    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Thinking...🤔"):
            result = qa_chain.invoke({"query": question})
            st.markdown("### 💡 Answer")
            st.write(result["result"])

            st.markdown("### Source Document(s)")
            for doc in result["source_documents"]:
                st.markdown(f"> {doc.page_content[:200]}...")