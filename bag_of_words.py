import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the CSV in the same folder
file_path = os.path.join(current_dir, "sentiment_data.csv")

# List of encodings to try
encodings_to_try = ['utf-8', 'cp1252', 'latin1', 'utf-8-sig']

# Step 1: Read CSV with multiple encoding attempts
for enc in encodings_to_try:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Successfully read CSV with encoding: {enc}")
        break
    except UnicodeDecodeError:
        continue
else:
    # Last resort: read with replacement of invalid characters
    print("All standard encodings failed. Reading CSV with 'utf-8' and replacing invalid chars.")
    df = pd.read_csv(file_path, encoding='utf-8', errors='replace')

print(df.head())

# Fill missing values
df["text"] = df["text"].fillna("")
df["sentiment"] = df["sentiment"].fillna("neutral")

# Function to clean text
def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    # Remove emojis and symbols (keep letters and numbers)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply cleaning
df["text"] = df["text"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

# Bag of Words vectorization
vectorizer = CountVectorizer(stop_words="english")
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_bow)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))