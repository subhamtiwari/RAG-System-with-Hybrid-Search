from nltk.tokenize import word_tokenize

def chunk_text(text, chunk_size=300,overlap=100):
    tokens = word_tokenize(text)
    chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size-overlap)]
    return chunks



