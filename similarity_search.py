import pdfplumber
from nltk.tokenize import sent_tokenize
import re
from sentence_transformers import SentenceTransformer, util


# Extract PDF text from PDF input
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = [page.extract_text() for page in pdf.pages]
        return ''.join(text)


# Preprocessing text by tokenizing the text into sentences
def preprocess_text(text):
    sentences = sent_tokenize(text)
    # Preprocess each sentence if needed (e.g., remove special characters, convert to lowercase)
    preprocessed_sentences = [re.sub(r'\W+', ' ', sentence.lower()) for sentence in sentences]
    return preprocessed_sentences


# Using BERT for similarity search
def search(query, sentences):
    # Initialize the SentenceTransformer model outside the search function
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])
    cosine_similarities = util.cos_sim(query_embedding, sentence_embeddings).flatten()
    op_results = sorted([(index, score.item()) for index, score in enumerate(cosine_similarities)], key=lambda x: x[1],
                        reverse=True)

    filtered_results = [(index, score) for index, score in op_results if len(sentences[index]) > 25]
    return filtered_results
