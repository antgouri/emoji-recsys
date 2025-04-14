# src/evaluations.py

import numpy as np
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
import scipy.sparse as sp

from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
import scipy.sparse as sp

def evaluate_model(model: LightFM, test_interactions: sp.coo_matrix, train_interactions: sp.coo_matrix = None,
                   item_features=None, k: int = 5) -> dict:
    metrics = {}
    metrics["AUC"] = auc_score(model, test_interactions, train_interactions=train_interactions,
                               item_features=item_features, num_threads=4).mean()
    metrics["Precision@5"] = precision_at_k(model, test_interactions, train_interactions=train_interactions,
                                            item_features=item_features, k=k, num_threads=4).mean()
    metrics["Recall@5"] = recall_at_k(model, test_interactions, train_interactions=train_interactions,
                                      item_features=item_features, k=k, num_threads=4).mean()
    return metrics


def print_metrics(metrics: dict):
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

