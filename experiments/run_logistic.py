#!/usr/bin/env python3
"""
Run Logistic Regression experiment with hyperparameter tuning.
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

    # Train with tuning (uses val for hyperparameter selection)
    pipeline, X_test, y_test = train_logistic_model(data_path, tune=True)
    
    # Final evaluation on held-out test set
    y_pred = pipeline.predict(X_test)
    acc, report = compute_metrics(y_test, y_pred)

    print("\\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    print(f"Final Test Accuracy: {acc:.4f}")
    print("\nFinal Test Classification Report:\n", report)

    save_results(acc, report, os.path.join(results_dir, "logistic_regression_results_comprehensive.txt"))

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