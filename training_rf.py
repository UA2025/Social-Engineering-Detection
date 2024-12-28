import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import nltk

nltk.download('stopwords')
print("nltk done")

# Preprocessing funct
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Lowercase
    if not isinstance(text, str):
        return ""  # Return an empty string or a placeholder if text is not a string
    text = re.sub(r'<.*?>', '', text)
    return text

#--------------------------------DATA LOADING AND PREPROCESSING--------------------------------------------------------
pickle_file_path = 'processed_data.pkl'
data = pd.read_pickle(pickle_file_path)
print(data)

#--------------------------------RANDOM FORREST AND CLASSIFICATION------------------------------------------------------
# extracting features from text
vectorizer = TfidfVectorizer()
X_text_features = vectorizer.fit_transform(data['cleaned_text'])
print("Length of text features:", len(X_text_features.toarray()))

# combining text features with sentiment analysis
X_features_df = pd.DataFrame(X_text_features.toarray()).reset_index(drop=True)
sentiment_df = data[['sentiment']].reset_index(drop=True)
print("NaN values in sentiment before concatenation:", sentiment_df.isnull().sum())

X_combined = pd.concat([X_features_df, sentiment_df], axis=1)
X_combined = X_combined.dropna()

data['Email_Type'] = data['Email_Type'].map({'Safe Email': 0, 'Phishing Email': 1})
y = data['Email_Type']
print("Length of X_combined:", len(X_combined))
print("Length of y:", len(y))

# TRAINING
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred))

# SAVING MODEL
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')