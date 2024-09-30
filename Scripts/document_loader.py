import os
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.schema import Document

# Function to load a PDF using LangChain's UnstructuredPDFLoader
def read_pdf_file(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()  # Returns a list of Document objects
    return documents

# Function to load a DOCX file using LangChain's UnstructuredWordDocumentLoader
def read_word_file(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()  # Returns a list of Document objects
    return documents

# Function to load a document based on the file extension using LangChain
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_word_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Function to load all documents from a directory and append them together
def load_documents_from_folder(folder_path):
    all_documents = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Load documents based on file extension
                documents = load_document(file_path)

                # Ensure the documents are in the correct format and append to the list
                all_documents.extend(documents)  # documents is a list of Document objects
                
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return all_documents
