from fastapi import FastAPI
from pydantic import BaseModel
from app.summarizer import generate_summary
from app.keywords import extract_keywords

app = FastAPI(title="Article Analysis API")

class ArticleRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(request: ArticleRequest):
    summary = generate_summary(request.text)
    return {"summary": summary}

@app.post("/keywords")
def keywords(request: ArticleRequest):
    keywords = extract_keywords(request.text)
    return {"keywords": keywords}

@app.post("/analyze")
def analyze(request: ArticleRequest):
    summary = generate_summary(request.text)
    keywords = extract_keywords(request.text)
    return {
        "summary": summary,
        "keywords": keywords
    }