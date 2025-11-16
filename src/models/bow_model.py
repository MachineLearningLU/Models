from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from ..data.load_data import load_csv_with_encodings
from ..data.preprocess import clean_text
from ..utils.metrics import compute_metrics

def train_bow_model(data_path, test_size=0.2, random_state=42):
    """
    Train Bag-of-Words + Naive Bayes model.
    Returns trained model, X_test, y_test.
    """
    df = load_csv_with_encodings(data_path)
    df["text"] = df["text"].fillna("")
    df["sentiment"] = df["sentiment"].fillna("neutral")
    df["text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=test_size, random_state=random_state
    )

    vectorizer = CountVectorizer(stop_words="english")
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_bow, y_train)

    return model, X_test_bow, y_test, vectorizer