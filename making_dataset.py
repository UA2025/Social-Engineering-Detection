import pandas as pd
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import nltk

nltk.download('stopwords')
print("nltk done")


# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Lowercase
    if not isinstance(text, str):
        return ""
    return text


# Sliding window sentiment analysis function
def sliding_window_sentiment_analysis(text, model, tokenizer, window_size=512, stride=256):
    tokens = tokenizer.tokenize(text)
    sentiments = []

    # Create windows of tokens
    for i in range(0, len(tokens), stride):
        window_tokens = tokens[i:i + window_size]
        if len(window_tokens) == 0:
            break

        # Convert tokens to input IDs and create tensor
        input_ids = tokenizer.convert_tokens_to_ids(window_tokens)
        inputs = torch.tensor([input_ids])

        # Get predictions from the model
        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs.logits

        # Get predicted class (0 for negative, 1 for positive)
        predicted_class = logits.argmax().item()
        sentiments.append(predicted_class)
        print(sentiments)
    return sentiments


# Load data and preprocess it
data = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\IS\first_300_mails.csv')
print("DATASET READ")
print(data.columns.tolist())
data = data.dropna(subset=['Email_Text'])

data['cleaned_text'] = data['Email_Text'].apply(preprocess_text)

# Load pre-trained DistilBERT model and tokenizer for sentiment analysis
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
print("LOADED MODEL")

# Perform sentiment analysis using sliding window approach
data['sentiment'] = data['cleaned_text'].apply(lambda x: sliding_window_sentiment_analysis(x, model, tokenizer))
print("LOADED SENTIMENTS")

# 0 neg 1 pos
# Flatten the list of lists into a single list if needed (optional)
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if any(pred == 1 for pred in x) else 0 if x else None)

print(data.shape)
data.to_pickle("processed_data.pkl")