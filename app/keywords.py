import yake

def extract_keywords(text, top_n=5):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=top_n)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, _ in keywords]
