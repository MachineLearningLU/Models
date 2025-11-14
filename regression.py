import os
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- NLTK Resource Download (run once) ---
# You may need to run these lines once manually in your terminal/environment
# if you haven't already:
# nltk.download('wordnet')
# nltk.download('stopwords')
# ------------------------------------------

lemmatizer = WordNetLemmatizer()
# Add common words that don't carry sentiment to the stop_words list
stop_words = set(stopwords.words('english'))

# Function to clean text (with lemmatization)
def clean_text(text: str) -> str:
    text = str(text).lower()  # Ensure text is string and lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text) # Remove mentions and hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # Keep only letters and spaces

    # Tokenize, lemmatize, and remove stop words
    word_list = []
    for word in text.split():
        if word not in stop_words:
            word_list.append(lemmatizer.lemmatize(word))
    
    text = " ".join(word_list)
    return re.sub(r"\s+", " ", text).strip() # Clean up extra spaces

# --- Data Loading ---
# Try to get the script's directory, fall back to current working directory
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd() # Fallback for environments like notebooks

file_path = os.path.join(current_dir, "sentiment_data.csv")
encodings_to_try = ['utf-8', 'cp1252', 'latin1', 'utf-8-sig']

df = None
for enc in encodings_to_try:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Successfully read CSV with encoding: {enc}")
        break
    except (UnicodeDecodeError):
        continue # Try next encoding
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        exit()

if df is None:
    print("All standard encodings failed. Reading CSV with 'utf-8' and replacing invalid chars.")
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"Failed to read file even with error replacement: {e}")
        exit()

print("Data loaded successfully:")
print(df.head())

# --- Preprocessing ---
# Drop rows where 'text' is missing, as they have no predictive value
df.dropna(subset=['text'], inplace=True)
df["sentiment"] = df["sentiment"].fillna("neutral") # Fill missing sentiments
print("\nApplying text cleaning...")
df["text"] = df["text"].apply(clean_text)
print("Cleaning complete.")

# --- Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], 
    df["sentiment"], 
    test_size=0.2, 
    random_state=42, 
    stratify=df["sentiment"] # Ensure class balance in splits
)

# --- Build Optimal Pipeline ---
# We are now hard-coding the best parameters found by GridSearchCV
print("\nBuilding optimal pipeline...")
optimal_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.9,     # From your best params
        min_df=3,     # From your best params
        ngram_range=(1, 2) # From your best params
    )),
    ('clf', LogisticRegression(
        C=1,          # From your best params
        max_iter=1000, 
        class_weight='balanced'
    ))
])

# --- Train the Optimal Model ---
print("Training the model on the training data...")
optimal_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- Evaluate with Best Model ---
print("\n--- Test Set Evaluation ---")
y_pred = optimal_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Example Prediction ---
print("\n--- Example Prediction Test ---")
test_text_positive = "This is an amazing product, I love it!"
test_text_negative = "I really hate this, it's the worst thing ever."
test_text_neutral = "The package arrived today."

# We must apply the SAME cleaning function
cleaned_positive = clean_text(test_text_positive)
cleaned_negative = clean_text(test_text_negative)
cleaned_neutral = clean_text(test_text_neutral)

# The pipeline handles vectorization and prediction
pred_pos = optimal_pipeline.predict([cleaned_positive])
pred_neg = optimal_pipeline.predict([cleaned_negative])
pred_neu = optimal_pipeline.predict([cleaned_neutral])

print(f"'{test_text_positive}' -> Cleaned: '{cleaned_positive}' -> Predicted: {pred_pos[0]}")
print(f"'{test_text_negative}' -> Cleaned: '{cleaned_negative}' -> Predicted: {pred_neg[0]}")
print(f"'{test_text_neutral}' -> Cleaned: '{cleaned_neutral}' -> Predicted: {pred_neu[0]}")