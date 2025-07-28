from keybert import KeyBERT

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def extract_keywords(text: str, top_k: int = 5):
    keywords = kw_model.extract_keywords(text, top_n=top_k)
    return [kw for kw, _ in keywords]
