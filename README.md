Instructions to Set Up:
Install the necessary libraries:

Run the following command to install the required libraries:

bash
Copy code
pip install langchain transformers sentence-transformers faiss-cpu openai pdfminer.six docx requests
Install the appropriate LLM locally (optional for LLaMA users):

Make sure the required local LLM (such as LLaMA) is set up using LM Studio or any preferred method.

Run the Code:

Copy the code below into a Python script (e.g., rag_project.py) and run it with python rag_project.py.

Complete RAG Project in a Single Python File:
python
Copy code
# Import necessary modules and libraries
import os
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import requests

# ---------- Step 1: Document Loading (PDFs and DOCX Files) ----------

def read_pdf_file(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

def read_word_file(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()
    return documents

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_word_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def load_documents_from_folder(folder_path):
    document_texts = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                documents = load_document(file_path)
                document_texts.extend(documents)
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return document_texts

# ---------- Step 2: Preprocessing and Chunking ----------

def preprocess_and_chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

# ---------- Step 3: Embeddings Generation ----------

# Generate SentenceTransformer embeddings
def generate_general_embeddings(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([doc.page_content for doc in documents])
    return embeddings

# ---------- Step 4: Faiss Vector Storage and Search ----------

def store_embeddings_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_from_faiss(query_embedding, index, top_k=5):
    query_vector = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return indices, distances

# ---------- Step 5: LLM-Based Response Generation Using LLaMA (or FLAN-T5) ----------

def generate_llama_response(prompt, model_name="LM Studio Community/Meta-Llama-3-7B-Instruct"):
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# ---------- Main Function to Execute the Entire RAG System ----------

def main():
    # Section 1: Load Documents from a Folder
    folder_path = input("Enter the path to the folder containing documents: ")
    documents = load_documents_from_folder(folder_path)
    if not documents:
        print("No documents loaded. Exiting...")
        return
    print(f"Loaded {len(documents)} documents.")

    # Section 2: Preprocess and Chunk the Documents
    chunked_documents = preprocess_and_chunk_documents(documents, chunk_size=500, chunk_overlap=100)
    print(f"Preprocessed and chunked the documents into {len(chunked_documents)} parts.")

    # Section 3: Generate Embeddings for Chunked Documents
    embeddings = generate_general_embeddings(chunked_documents)
    print(f"Generated {len(embeddings)} embeddings.")

    # Section 4: Store Embeddings in Faiss
    faiss_index = store_embeddings_in_faiss(embeddings)
    print("Stored embeddings in Faiss index.")

    # Section 5: Retrieve Documents Using Faiss
    query = input("Enter your query: ")
    query_embedding = generate_general_embeddings([{"page_content": query}])[0]  # Single query embedding
    retrieved_indices, distances = retrieve_from_faiss(query_embedding, faiss_index, top_k=5)
    print(f"Retrieved indices: {retrieved_indices}")
    print(f"Similarity distances: {distances}")

    # Section 6: Construct Combined Context for Generation
    combined_context = " ".join([chunked_documents[idx].page_content for idx in retrieved_indices[0]])
    print(f"Combined Context:\n{combined_context[:500]}")  # Print the first 500 characters of the context

    # Section 7: Generate Response Using LLaMA Model (or Flan-T5)
    response = generate_llama_response(query, combined_context)
    print(f"Generated Response:\n{response}")

if __name__ == "__main__":
    main()
Key Features:
Document Loading:

Supports multiple document formats such as PDFs and DOCX using LangChain.
Document Chunking:

Breaks down documents into manageable chunks using RecursiveCharacterTextSplitter.
Embedding Generation:

Uses SentenceTransformers for embedding generation.
Hybrid Search:

Uses Faiss for semantic search. (Elasticsearch can be added later if needed).
Generative AI Response:

Uses locally hosted LLaMA model via HTTP API or Google’s FLAN-T5 for response generation.
How to Run:
Set up the Python environment with the required packages.
Download a local LLM server (optional if you want to use Flan-T5 or OpenAI).
Start the script, and follow the interactive prompts to upload documents and query the system.
Feel free to let me know if you'd like to further refine or expand any specific functionality!






