from sklearn.metrics import accuracy_score, classification_report

def compute_metrics(y_true, y_pred):
    """
    Compute accuracy and classification report.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return acc, report

def save_results(acc, report, filepath):
    """
    Save accuracy and report to file.
    """
    with open(filepath, 'w') as f:
        f.write(f"Accuracy: {acc}\n\nClassification Report:\n{report}")