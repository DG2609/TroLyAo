# rag_system.py
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from module.embedder import get_custom_embedder  # Import the custom embedder
from huggingface_hub import login
import re

login("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")


def extract_document_number(text):
    match = re.search(r'\d{2,4}/\d{4}/[A-ZĐ]-[A-Z]+', text)
    return match.group(0) if match else None

def build_faiss_index(documents):
    # Initialize embeddings using the tokenizer
    embedder = get_custom_embedder()
    
    # Define the embedding function
    def embedding_function(text):
        return embedder.encode([text])[0]
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    metadatas = []
    
    for doc in documents:
        doc_number = extract_document_number(doc['text'])
        splits = text_splitter.split_text(doc['text'])
        texts.extend(splits)
        metadatas.extend([{'source': doc['file_name'], 'doc_number': doc_number}] * len(splits))
        
    embeddings = embedder.encode(texts)
    text_embeddings = list(zip(texts, embeddings))
    # Build the FAISS index
    vectorstore = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embedder, metadatas=metadatas)
    vectorstore.embedding_function = embedding_function  # Set the embedding function separately

    return vectorstore