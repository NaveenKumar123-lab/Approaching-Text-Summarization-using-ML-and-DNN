from django.shortcuts import render
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from spacy.lang.en import English
import numpy as np

def summarize(data):
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = nlp(data.replace("\n", ""))
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # TF-IDF Vectorization
    tf_idf_vectorizer = TfidfVectorizer(smooth_idf=True)
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(sentences)
    
    # Apply LSA for dimensionality reduction
    n_components = min(100, tf_idf_matrix.shape[1])  # Ensure n_components is not greater than number of features
    lsa = TruncatedSVD(n_components=n_components)
    lsa_sentences = lsa.fit_transform(tf_idf_matrix)
    
    # Calculate sentence scores
    sentence_scores = np.sum(lsa_sentences, axis=1)
    
    # Select top N sentences as summary
    N = 4
    top_n_indices = np.argsort(sentence_scores)[::-1][:N]
    summary_sentences = [sentences[idx] for idx in top_n_indices]
    summary = " ".join(summary_sentences)
    
    return {"original_text": data, "summary": summary}

def home(request):
    data = {}
    if request.POST:
        data = summarize(request.POST['text'])
    return render(request, "home.html", data)
