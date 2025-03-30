# text-summarizer

 Text Summarizer API  

This project is a Flask-based API that provides text summarization using three different NLP techniques:  

1. NLTK-Based Summarization (Frequency-Based)  
   - Tokenizes the text into words and sentences.  
   - Calculates word frequency and scores each sentence based on word occurrence.  
   - Selects the most relevant sentences for the summary.  

2. SpaCy-Based Summarization (NER-Based)  
   - Uses Named Entity Recognition (NER) to rank sentences.  
   - Prioritizes sentences containing important entities (e.g., names, locations).  

3. Word2Vec-Based Summarization (Semantic Similarity)  
   - Trains a Word2Vec model on the input text.  
   - Computes sentence embeddings and selects key sentences based on similarity with the overall document.  

 Technologies Used:
- Flask (Backend API)  
- Flask-CORS (For frontend-backend communication)  
- **NLTK** (Tokenization, word frequency analysis)  
- SpaCy (Named Entity Recognition)  
- Gensim Word2Vec (Word embeddings for semantic summarization)  
- Scikit-Learn (Cosine similarity for sentence ranking)  

Project Objective:  
To provide an efficient text summarization tool that allows users to generate summaries using different NLP techniques. 

