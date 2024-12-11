# server.py
from flask import Flask, request, jsonify
from module.pdf_loader import extract_text_from_pdfs
from module.tokenizer_module import tokenize_documents
from module.rag_system import build_faiss_index
from module.model_loader import load_llama_model
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipelines
from module.embedder import get_custom_embedder  # Import the custom embedder
import re
import torch

app = Flask(__name__)

folder_path = "./data"

# Load and extract text from PDFs
documents, error_files = extract_text_from_pdfs(folder_path)

# Check for errors
if error_files:
    print("\nFiles with errors:")
    for error_file, reason in error_files:
        print(f"{error_file}: {reason}")
else:
    print("All files processed successfully.")

# Build FAISS index using the documents
vectorstore = build_faiss_index(documents)

# Load the Llama model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize HuggingFace Pipeline
pipe = pipelines.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=200,
)

# Initialize HuggingFacePipeline for LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Generate answer
    full_answer = qa.run(question)

        # Extract the response after "Helpful Answer:"
    if "Helpful Answer:" in full_answer:
        answer = full_answer.split("Helpful Answer:")[1].strip()
    else:
        answer = full_answer.strip()
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, use_reloader=False)