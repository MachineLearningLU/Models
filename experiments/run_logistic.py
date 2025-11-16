#!/usr/bin/env python3
"""
Run Logistic Regression experiment.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.logistic_model import train_logistic_model
from src.utils.metrics import compute_metrics, save_results

def main():
    data_path = "data/raw/sentiment_data.csv"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    pipeline, X_test, y_test = train_logistic_model(data_path)
    y_pred = pipeline.predict(X_test)
    acc, report = compute_metrics(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)

    save_results(acc, report, os.path.join(results_dir, "logistic_regression_results.txt"))

    # Example predictions
    test_texts = [
        "This is an amazing product, I love it!",
        "I really hate this, it's the worst thing ever.",
        "The package arrived today."
    ]

    for text in test_texts:
        pred = pipeline.predict([text])
        print(f"'{text}' -> Predicted: {pred[0]}")

if __name__ == "__main__":
    main()