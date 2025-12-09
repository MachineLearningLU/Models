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
    # Set use_random_search=True for faster RandomizedSearchCV (recommended for large grids)
    model, X_test, y_test = train_bow_model(data_path, tune=True, use_random_search=False)
    
    # Final evaluation on held-out test set
    y_pred = model.predict(X_test)
    acc, report = compute_metrics(y_test, y_pred)

    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    print(f"Final Test Accuracy: {acc:.4f}")
    print("\nFinal Test Classification Report:\n", report)

    save_results(acc, report, os.path.join(results_dir, "bag_of_words_results_comprehensive.txt"))

if __name__ == "__main__":
    main()