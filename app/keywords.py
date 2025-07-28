from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords(text, max_keywords=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    top_indices = tfidf_scores.argsort()[::-1][:max_keywords]
    top_keywords = feature_array[top_indices]

    return top_keywords.tolist()
