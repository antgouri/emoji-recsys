# src/baseline.py

import pandas as pd
import json
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from sklearn.model_selection import train_test_split

def build_baseline_model(df: pd.DataFrame):
    dataset = Dataset()
    dataset.fit(df["user_id"], df["product_id"])

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    train_interactions, train_weights = dataset.build_interactions([
        (row["user_id"], row["product_id"], row["rating"]) for _, row in train_df.iterrows()
    ])

    test_interactions, _ = dataset.build_interactions([
        (row["user_id"], row["product_id"], row["rating"]) for _, row in test_df.iterrows()
    ])

    model = LightFM(loss='logistic',random_state=42)
    model.fit(train_interactions, sample_weight=train_weights, epochs=30, num_threads=1)

    return model, train_interactions, test_interactions

def evaluate_baseline(model, train_interactions, test_interactions):
    metrics = {}
    metrics["AUC"] = auc_score(model, test_interactions, train_interactions=train_interactions, num_threads=4).mean()
    metrics["Precision@5"] = precision_at_k(model, test_interactions, train_interactions=train_interactions, k=5, num_threads=4).mean()
    metrics["Recall@5"] = recall_at_k(model, test_interactions, train_interactions=train_interactions, k=5, num_threads=4).mean()
    return metrics

def print_metrics(metrics: dict):
    print("\nBaseline Evaluation Results (No Emoji / No Features):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

def run_baseline_pipeline(json_path: str):
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df = df.rename(columns={
        'reviewerID': 'user_id',
        'asin': 'product_id',
        'overall': 'rating',
        'reviewText': 'review_text'
    })
    df = df[['user_id', 'product_id', 'rating', 'review_text']].dropna()

    print("Training baseline LightFM model (no features)...")
    model, train_interactions, test_interactions = build_baseline_model(df)
    metrics = evaluate_baseline(model, train_interactions, test_interactions)
    print_metrics(metrics)

