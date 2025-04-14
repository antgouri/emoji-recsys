##  Emoji-Infused Hybrid Recommender System with evaluation and explainability 

This repository contains a modular Python implementation of a hybrid recommender system enhanced with emoji-based sentiment enrichment and SHAP explanations. It also includes evaluation metrics like AUC, Precision@5, and Recall@5. 
This work is prepared for submission to ACM RecSys 2025 as a short paper.

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/<your_username>/emoji-recsys.git
cd emoji-recsys

### 2. Create a Conda Environment and Install Requirements (I prefer to use conda environments only)

conda create -n emosys python==3.10
conda activate emosys

pip install -r requirements.txt
python -m textblob.download_corpora

## Run the Full Pipeline

python scripts/run_pipeline.py

This will:
1. Load and preprocess the data from Musical_Instruments_5.json.
2. Inject sentiment-based emojis to the reviews.
3. Train a LightFM hybrid recommendation model.
4. Generate SHAP explanations and save plots to outputs/.
5. Evaluate the model using AUC, Precision@5, and Recall@5.

## Output Artifacts

| Artifact | Location |
|---------|----------|
| Emoji-injected reviews | csv/emoji_reviews.csv |
| SHAP summary plot | outputs/shap_summary.png |
| SHAP raw values | outputs/shap_values.npy |
| Evaluation results | Printed to terminal |

## Evaluation Metrics

The recommender is evaluated using:
- AUC (Area Under Curve): How well the model ranks positive vs negative items.
- Precision@5: Proportion of top-5 recommended items that are relevant.
- Recall@5: Proportion of all relevant items recommended in top-5.

These are computed using:
from src.evaluations import evaluate_model, print_metrics

## Dataset Used

- Source: [Amazon Musical Instruments 5-core dataset](https://nijianmo.github.io/amazon/index.html)
- Format: JSON
- Fields Used: reviewerID, asin, overall, reviewText

## Dependencies

Listed in requirements.txt:
- lightfm, textblob, pandas, scikit-learn, shap, matplotlib, nltk, cupy-cuda12x (for GPU support)

Install them with:

pip install -r requirements.txt


## Author

Developed by [Dr. Ananth G S and Dr. K. Raghuveer] as part of short paper work for ACM RecSys 2025.

## Feedback & Contributions

Feel free to open issues or submit pull requests to improve sentiment detection, support new explainability methods, or add datasets!

> Star this repo if you find it useful or insightful!
