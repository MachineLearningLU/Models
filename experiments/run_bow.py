#!/usr/bin/env python3
"""
Run Bag-of-Words experiment.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.bow_model import train_bow_model
from src.utils.metrics import compute_metrics, save_results

def main():
    data_path = "data/raw/sentiment_data.csv"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    model, X_test, y_test, vectorizer = train_bow_model(data_path)
    y_pred = model.predict(X_test)
    acc, report = compute_metrics(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)

    save_results(acc, report, os.path.join(results_dir, "bag_of_words_results.txt"))

if __name__ == "__main__":
    main()

def main():
    data_path = "data/raw/sentiment_data.csv"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    model, X_test, y_test, vectorizer = train_bow_model(data_path)
    y_pred = model.predict(X_test)
    acc, report = compute_metrics(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)

    save_results(acc, report, os.path.join(results_dir, "bag_of_words_results.txt"))

if __name__ == "__main__":
    main()