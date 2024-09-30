import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Uncomment to download necessary NLTK resources if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to lowercase text
def lowercase_text(text):
    return text.lower()

# Function to remove punctuation while keeping special characters like $ and %
def remove_punctuation(text):
    return re.sub(r'[^\w\s\$\%\€]', '', text)

# Function to remove stopwords using NLTK
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Function to lemmatize the text using WordNet Lemmatizer
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

# Core preprocessing pipeline for LangChain Document objects
def preprocess_text(text):
    """
    Applies a series of preprocessing steps to the input text:
    - Lowercase conversion
    - Punctuation removal
    - Stopword removal
    - Lemmatization
    """
    text = lowercase_text(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# Function to preprocess and chunk the documents
def preprocess_and_chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Preprocesses and chunks a list of LangChain Document objects.

    Args:
    - documents: List of LangChain Document objects to preprocess and chunk.
    - chunk_size: Number of characters in each chunk.
    - chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
    - List of preprocessed and chunked documents.
    """
    preprocessed_docs = []

    for idx, doc in enumerate(documents):
        try:
            # Ensure the document has the 'page_content' field
            if not hasattr(doc, 'page_content'):
                raise ValueError(f"Document at index {idx} does not contain 'page_content'.")

            # Preprocess the content
            preprocessed_text = preprocess_text(doc.page_content)
            
            # Create a new Document object with preprocessed content, keeping original metadata
            preprocessed_doc = Document(page_content=preprocessed_text, metadata=doc.metadata)
            preprocessed_docs.append(preprocessed_doc)

        except Exception as e:
            print(f"Error processing document at index {idx}: {e}")

    # Use LangChain's RecursiveCharacterTextSplitter to split preprocessed documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(preprocessed_docs)

    return chunked_docs
