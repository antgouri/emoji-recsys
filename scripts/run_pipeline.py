# scripts/run_pipeline.py

import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random

from src.preprocessing import load_amazon_reviews
from src.emoji_injection import inject_emojis_to_reviews, save_emoji_reviews, compute_item_sentiment_features
from src.model_lightfm import build_and_train_model
from src.explainability import explain_model_with_shap
from src.evaluations import evaluate_model, print_metrics
from src.baseline import run_baseline_pipeline
from src.improved import run_tfidf_pipeline
from src.improved_model import run_lda_pipeline

def set_global_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_global_seed(42)

DATA_PATH = "data/Musical_Instruments_5.json"
EMOJI_CSV_PATH = "csv/emoji_reviews.csv"
OUTPUT_DIR = "outputs"

def run_emoji_model():
    print("Phase 1: Loading and preprocessing data...")
    df = load_amazon_reviews(DATA_PATH, max_reviews=10000)

    print("Phase 2: Injecting sentiment-based emojis into reviews...")
    df_with_emojis = inject_emojis_to_reviews(df)
    save_emoji_reviews(df_with_emojis, EMOJI_CSV_PATH)

    print("Phase 3: Computing item sentiment features...")
    item_features_df = compute_item_sentiment_features(df_with_emojis)

    print("Phase 4: Building and training hybrid LightFM model with sentiment features...")
    model, dataset, train_df, test_df, test_interactions, item_features = build_and_train_model(
        df_with_emojis, item_features_df
    )

    print("Phase 5: Explaining model predictions using SHAP...")
    explain_model_with_shap(model, test_df, dataset, OUTPUT_DIR)

    print("Phase 6: Evaluating model on test set...")
    metrics = evaluate_model(model, test_interactions, item_features=item_features)
    print_metrics(metrics)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_pipeline.py [baseline | emoji_model | tfidf_model | lda_model]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "baseline":
        run_baseline_pipeline(DATA_PATH)
    elif mode == "emoji_model":
        run_emoji_model()
    elif mode == "tfidf_model":
        run_tfidf_pipeline(DATA_PATH)
    elif mode == "lda_model":
        run_lda_pipeline(DATA_PATH)
        
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python3 pipeline.py [baseline | emoji_model | tfidf_model | lda_model]")
        sys.exit(1)
        
if __name__ == "__main__":
    main()

