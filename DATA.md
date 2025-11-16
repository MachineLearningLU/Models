# Datasets Documentation

## Primary Dataset: sentiment_data.csv
- **Location**: `data/raw/sentiment_data.csv`
- **Size**: ~241,147 rows
- **Columns**: `text` (str), `sentiment` (int: 0=negative, 1=neutral, 2=positive)
- **Source**: [Kaggle Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset)
- **Notes**: Labels are numeric; mapping inferred from sample rows. Contains social media text (likely Twitter). Check for privacy: if text includes user IDs, do not redistribute without consent.

## Additional Datasets
Located in `data/raw/datasets/` (zipped to save space).
- **Sentiment Analysis.zip**: From [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis)
- **sentiment_analysis_dataset.zip**: From [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset)
- **sentiment_analysis_dataset012.zip**: From [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- **twitter_sentiment_analysis.zip**: From [Kaggle Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **twitter_tweets_sentiment_dataset.zip**: From [Kaggle Twitter Tweets Sentiment Dataset](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)

To unpack: Run `python scripts/unpack_datasets.py`. Contents will be in `data/raw/unpacked/`.

## Usage Notes
- Use `data/processed/` for cleaned/split data.
- For large datasets, consider sampling (e.g., 32k rows) for quick experiments.
- Provenance: All from Kaggle; check licenses for redistribution.