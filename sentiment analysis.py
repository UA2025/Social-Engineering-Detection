import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
import numpy as np
import joblib
import torch
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




#--------------------------------MODEL  LOADING--------------------------------------------------------
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = joblib.load('random_forest_model.pkl')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
sent_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

with open('tfidf_vectorizer.pkl', 'rb') as file2:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
print("LOADED MODELs")



def main():
    user_input = input("Enter mail text: ")
    print(f"You entered: {user_input}")
    cleaned_text = preprocess_text(user_input)
    sentiments = sliding_window_sentiment_analysis(cleaned_text,sent_model,tokenizer)
    sentiment_value = 1 if any(s == 1 for s in sentiments) else 0
    email_features = vectorizer.transform([cleaned_text]).toarray()
    combined_features = np.hstack((email_features, np.array([[sentiment_value]])))
    prediction = rf_model.predict(combined_features)
    result = "Phishing Email" if prediction[0] == 1 else "Safe Email"

    print(f"The classified result is: {result}")

if __name__ == "__main__":
    main()