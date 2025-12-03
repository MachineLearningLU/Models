# Social Media Reply Categorizer

This project automatically categorizes replies or comments on social media posts into categories like positive, negative, rude, spam, neutral, etc. It uses natural language processing (NLP) models trained on labeled datasets to analyze text and predict categories, helping with moderation and insights.

## Team Members
- Anthony Viglielmo
- Berghan Barnitz
- Emilio Vilchis
- Rudra Patel

## Dataset
Primary dataset: [Kaggle Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset)  
Additional datasets in `data/raw/datasets/` (zipped; unpack with `python scripts/unpack_datasets.py`).

## Installation
1. Clone the repo.
2. Create a virtual environment: `python -m venv myenv` (or use existing).
3. Activate: `source myenv/Scripts/activate` (Windows bash).
4. Install deps: `pip install -r requirements.txt`.
5. (Optional for transformers): `pip install -r requirements-roberta.txt`.

## Running Experiments
The project uses a modular structure for reproducible experiments.

### Sentiment Classification Models
- **Bag-of-Words + Naive Bayes**: `python experiments/run_bow.py`
- **TF-IDF + Logistic Regression**: `python experiments/run_logistic.py`

Both scripts:
- Load data from `data/raw/sentiment_data.csv`
- Perform hyperparameter tuning on validation set (60/20/20 split)
- Evaluate on held-out test set
- Save results to `results/` directory
- Print accuracy, classification report, and example predictions

Expected results:
- BOW: ~66% test accuracy
- Logistic: ~79% test accuracy

### Custom Experiments
Modify `experiments/config/` YAML files for different hyperparameter grids, then run the scripts.


## Project Structure
- `data/`: Raw and processed datasets.
- `src/`: Reusable code modules.
- `experiments/`: Scripts to run models.
- `notebooks/`: Jupyter notebooks for analysis.
- `models/`: Saved model artifacts.
- `results/`: Experiment outputs.
- `scripts/`: Helper scripts.
- `docs/`: Documentation.
- `tests/`: Unit tests.

## Goals
- Train effective NLP models on small samples.
- Expand to contextual sentiment (post + reply).
- Add spam and inappropriate content detection.