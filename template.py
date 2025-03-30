from flask import Flask, request, jsonify
import spacy
import nltk
import re
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from collections import Counter

# Download necessary resources
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

def clean_text(text):
    """Preprocess text by removing tags, URLs, and extra spaces."""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

def nltk_summarizer(text, ratio=0.3):
    """Summarization using sentence frequency."""
    sentences = sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    word_freq = Counter(words)  # Frequency count of words
    
    sentence_scores = {}
    for sentence in sentences:
        sentence_scores[sentence] = sum(word_freq[word] for word in nltk.word_tokenize(sentence.lower()) if word in word_freq)
    
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    num_sentences = max(1, int(len(sentences) * ratio))
    
    return ' '.join(sorted_sentences[:num_sentences])

def spacy_summarizer(text, ratio=0.3):
    """Summarization using Named Entity Recognition (NER)."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Give importance to sentences containing named entities
    sentence_scores = {}
    for sent in sentences:
        sentence_scores[sent] = sum(1 for ent in nlp(sent).ents)  # Count named entities
    
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    num_sentences = max(1, int(len(sentences) * ratio))
    
    return ' '.join(sorted_sentences[:num_sentences])

def train_word2vec(text):
    """Train Word2Vec model and extract key sentences based on similarity."""
    sentences = [sent.split() for sent in sent_tokenize(text)]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # Compute sentence embeddings by averaging word vectors
    sentence_embeddings = []
    for sent in sentences:
        vectors = [model.wv[word] for word in sent if word in model.wv]
        if vectors:
            sentence_embeddings.append(np.mean(vectors, axis=0))
        else:
            sentence_embeddings.append(np.zeros(100))
    
    # Compute similarity of each sentence with the overall text
    overall_embedding = np.mean(sentence_embeddings, axis=0)
    similarities = [cosine_similarity([sent_emb], [overall_embedding])[0][0] for sent_emb in sentence_embeddings]

    # Sort sentences by similarity score
    sorted_sentences = [sent for _, sent in sorted(zip(similarities, sent_tokenize(text)), reverse=True)]
    num_sentences = max(1, int(len(sentences) * 0.3))

    return ' '.join(sorted_sentences[:num_sentences])

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text', '').strip()
    method = data.get('method', '').strip().lower()  # Ensure method is lowercase and trimmed
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if method not in ['nltk', 'spacy', 'word2vec']:
        return jsonify({'error': 'Invalid summarization method'}), 400  # Strict validation

    cleaned_text = clean_text(text)

    # Ensure each method produces different results
    if method == 'nltk':
        summary = nltk_summarizer(cleaned_text)
    elif method == 'spacy':
        summary = spacy_summarizer(cleaned_text)
    elif method == 'word2vec':
        summary = train_word2vec(cleaned_text)

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
