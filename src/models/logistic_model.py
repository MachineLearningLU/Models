from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import yaml
import os
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

def train_logistic_model(data_path, tune=False, random_state=42):
    """
    Train TF-IDF + LogisticRegression model with optional hyperparameter tuning.
    Uses 60/20/20 train/val/test split. If tune=True, tunes on val and returns best model.
    
    Args:
        data_path: Path to the CSV data file
        tune: Whether to perform hyperparameter tuning
        random_state: Random seed for reproducibility
    
    Returns:
        Trained pipeline, X_test, y_test
    """
    df = load_csv_with_encodings(data_path)
    df.dropna(subset=['text'], inplace=True)
    df["sentiment"] = df["sentiment"].fillna("neutral")
    df["text"] = df["text"].apply(clean_text_lemmatize)

    # 60/20/20 split: train (60%), val (20%), test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=random_state, stratify=df["sentiment"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp  # 0.25 of 80% = 20%
    )

    if tune:
        # Hyperparameter tuning using pipeline-level grid search
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'config', 'logistic_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract cv_folds from config, default to 5
        cv_folds = config.pop('cv_folds', 5)
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression(class_weight='balanced'))
        ])
        
        # GridSearchCV across entire pipeline
        search = GridSearchCV(
            pipeline, config, cv=cv_folds,
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        print(f"Using GridSearchCV with {cv_folds}-fold CV")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")
        
        # Evaluate on val for development
        y_val_pred = model.predict(X_val)
        val_acc, val_report = compute_metrics(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("Validation Report:\n", val_report)
    else:
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_df=0.9, min_df=3, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(C=1, max_iter=1000, class_weight='balanced'))
        ])
        model = pipeline.fit(X_train, y_train)

    return model, X_test, y_test