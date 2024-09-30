import os
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to load a PDF using LangChain's UnstructuredPDFLoader
def load_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

# Function to load a DOCX file using LangChain's UnstructuredWordDocumentLoader
def load_docx(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()
    return documents

# Function to load a document based on the file extension using LangChain
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext == '.docx':
        return load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Function to load all documents from a directory and append them together
def load_documents_from_folder(folder_path):
    document_texts = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                documents = load_document(file_path)
                document_texts.extend(documents)  # Append documents
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return document_texts

# Function to streamline and split documents into chunks for better querying
def process_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    return split_docs


