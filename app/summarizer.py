from gradio_client import Client

client = Client("talalam23/summarization-model")

def generate_summary(text: str) -> str:
    try:
        result = client.predict(
            text=text,
            api_name="/predict"
        )
        return result
    except Exception as e:
        return f"Request failed: {str(e)}"
