#SHAP Explainability with images saved to output
import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.preprocessing import LabelEncoder

def explain_model_with_shap(model: LightFM, df: pd.DataFrame, dataset: Dataset, output_dir: str = "outputs"):
    
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Encode user and item IDs
    user_encoder = LabelEncoder().fit(df["user_id"])
    item_encoder = LabelEncoder().fit(df["product_id"])
    df["user_encoded"] = user_encoder.transform(df["user_id"])
    df["item_encoded"] = item_encoder.transform(df["product_id"])

    # Step 2: Prepare input matrix for SHAP
    X = df[["user_encoded", "item_encoded"]].to_numpy()

    # Step 3: Define prediction function
    def predict_lightfm(X_batch):
        user_ids = X_batch[:, 0].astype(int)
        item_ids = X_batch[:, 1].astype(int)
        return model.predict(user_ids, item_ids)

    # Step 4: SHAP KernelExplainer
    explainer = shap.KernelExplainer(predict_lightfm, X[:100])
    shap_values = explainer.shap_values(X[:50], nsamples=100)

    # Step 5: Save outputs
    shap.summary_plot(shap_values, features=X[:50], feature_names=["user_encoded", "item_encoded"], show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    np.save(os.path.join(output_dir, "shap_values.npy"), shap_values)
    np.save(os.path.join(output_dir, "shap_input.npy"), X[:50])

