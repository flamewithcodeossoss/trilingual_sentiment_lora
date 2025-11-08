# api.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel

# Load model and tokenizer once (global)
BASE_MODEL = "FacebookAI/xlm-roberta-base"
ADAPTER_MODEL = "osamanaguib/trilingual-sentiment-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model.eval()

# Create a pipeline for sentiment analysis
sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

# Label map
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

def analyze_sentiment(text: str):
    """Takes a text input and returns predicted sentiment and scores."""
    result = sentiment_pipeline(text)[0]
    # Get label with max score
    best = max(result, key=lambda x: x['score'])
    sentiment = label_map.get(best['label'], best['label'])
    return {
        "input": text,
        "sentiment": sentiment,
        "scores": {
            label_map.get(r['label'], r['label']): round(r['score'], 4)
            for r in result
        }
    }