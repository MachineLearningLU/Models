from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from ..data.load_data import load_csv_with_encodings
from ..data.preprocess import clean_text
from ..utils.metrics import compute_metrics

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_lemmatize(text: str) -> str:
    """
    Clean and lemmatize text.
    """
    text = clean_text(text)
    word_list = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(word_list)

def train_logistic_model(data_path, test_size=0.2, random_state=42):
    """
    Train TF-IDF + LogisticRegression model.
    Returns trained pipeline, X_test, y_test.
    """
    df = load_csv_with_encodings(data_path)
    df.dropna(subset=['text'], inplace=True)
    df["sentiment"] = df["sentiment"].fillna("neutral")
    df["text"] = df["text"].apply(clean_text_lemmatize)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=test_size, random_state=random_state, stratify=df["sentiment"]
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.9, min_df=3, ngram_range=(1, 2))),
        ('clf', LogisticRegression(C=1, max_iter=1000, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test