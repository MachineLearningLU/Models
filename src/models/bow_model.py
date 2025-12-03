from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from ..data.load_data import load_csv_with_encodings
from ..data.preprocess import clean_text
from ..utils.metrics import compute_metrics

def train_bow_model(data_path, tune=False, random_state=42):
    """
    Train Bag-of-Words + Naive Bayes model with optional hyperparameter tuning.
    Uses 60/20/20 train/val/test split. If tune=True, tunes on val and returns best model.
    Returns trained model, X_test, y_test, vectorizer.
    """
    df = load_csv_with_encodings(data_path)
    df["text"] = df["text"].fillna("")
    df["sentiment"] = df["sentiment"].fillna("neutral")
    df["text"] = df["text"].apply(clean_text)

    # 60/20/20 split: train (60%), val (20%), test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=random_state, stratify=df["sentiment"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp  # 0.25 of 80% = 20%
    )

    vectorizer = CountVectorizer(stop_words="english")
    X_train_bow = vectorizer.fit_transform(X_train)
    X_val_bow = vectorizer.transform(X_val)
    X_test_bow = vectorizer.transform(X_test)

    if tune:
        # Hyperparameter tuning on train, scored on val
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        grid = GridSearchCV(MultinomialNB(), param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train_bow, y_train)
        model = grid.best_estimator_
        print(f"Best val params: {grid.best_params_}")
        # Evaluate on val for development
        y_val_pred = model.predict(X_val_bow)
        val_acc, val_report = compute_metrics(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_acc}")
        print("Validation Report:\n", val_report)
    else:
        model = MultinomialNB()
        model.fit(X_train_bow, y_train)

    return model, X_test_bow, y_test, vectorizer