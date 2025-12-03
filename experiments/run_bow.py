#!/usr/bin/env python3
"""
Run Bag-of-Words experiment with hyperparameter tuning.
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

    # Train with tuning (uses val for hyperparameter selection)
    model, X_test, y_test, vectorizer = train_bow_model(data_path, tune=True)
    
    # Final evaluation on held-out test set
    y_pred = model.predict(X_test)
    acc, report = compute_metrics(y_test, y_pred)

    print("Final Test Accuracy:", acc)
    print("\nFinal Test Classification Report:\n", report)

    save_results(acc, report, os.path.join(results_dir, "bag_of_words_results.txt"))

if __name__ == "__main__":
    main()