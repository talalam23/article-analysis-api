from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Hugging Face model name
MODEL_NAME = "talalam23/t5-small-summarizer-dailymail"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_summary(text: str) -> str:
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
