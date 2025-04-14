#Hybrid lightFM model - currently using the warp loss function
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

try:
    import cupy as cp
    from lightfm.cuda_extensions import get_lightfm_cuda_module
    GPU_ENABLED = True
except ImportError:
    cp = np
    GPU_ENABLED = False

def build_interaction_matrices(df: pd.DataFrame, user_col="user_id", item_col="product_id", rating_col="rating"):
    dataset = Dataset()
    dataset.fit(df[user_col], df[item_col])
    
    (interactions, weights) = dataset.build_interactions(
        [(row[user_col], row[item_col], row[rating_col]) for _, row in df.iterrows()]
    )
    return dataset, interactions, weights

def train_lightfm_model(interactions, weights, loss='warp', epochs=30, num_threads=4):
    model = LightFM(loss=loss, no_components=32)
    model.fit(interactions, sample_weight=weights, epochs=epochs, num_threads=num_threads)
    return model

def build_and_train_model(df: pd.DataFrame, item_features_df: pd.DataFrame):
    dataset = Dataset()
    dataset.fit(df["user_id"], df["product_id"],
                item_features=[str(s) for s in item_features_df["avg_sentiment"]])

    # Build mapping
    dataset.fit_partial(df["user_id"], df["product_id"],
                        item_features=[str(s) for s in item_features_df["avg_sentiment"]])

    # Build item feature matrix
    item_feature_tuples = [
        (row["product_id"], [str(row["avg_sentiment"])]) for _, row in item_features_df.iterrows()
    ]
    item_features = dataset.build_item_features(item_feature_tuples)

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    train_interactions, train_weights = dataset.build_interactions(
        [(row["user_id"], row["product_id"], row["rating"]) for _, row in train_df.iterrows()]
    )

    test_interactions, _ = dataset.build_interactions(
        [(row["user_id"], row["product_id"], row["rating"]) for _, row in test_df.iterrows()]
    )

    model = LightFM(loss='logistic', random_state=42)
    model.fit(train_interactions, sample_weight=train_weights, item_features=item_features, epochs=30, num_threads=1)

    return model, dataset, train_df, test_df, test_interactions, item_features

