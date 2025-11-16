import re
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text: str) -> str:
    """
    Clean text: lowercase, remove URLs, mentions, hashtags, non-alphanumeric, extra spaces.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics(y_true, y_pred):
    """
    Compute and return accuracy and classification report.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return acc, report