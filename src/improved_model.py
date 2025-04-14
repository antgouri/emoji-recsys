
import pandas as pd
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(filepath, max_reviews=None):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for i, line in enumerate(f) if not max_reviews or i < max_reviews]
    df = pd.DataFrame(data)
    df = df.rename(columns={'reviewerID': 'user_id', 'asin': 'product_id', 'overall': 'rating', 'reviewText': 'review_text'})
    df = df[['user_id', 'product_id', 'rating', 'review_text']].dropna()
    return df

def extract_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.findall(text)

def cluster_emojis(df, n_clusters=5):
    df = df.copy()
    df['emojis'] = df['review_text'].apply(lambda x: extract_emojis(str(x)))
    df['emoji_str'] = df['emojis'].apply(lambda x: ' '.join(x) if x else '')

    if df['emoji_str'].str.len().sum() == 0:
        df['emoji_cluster'] = 'none'
        return df[['product_id', 'emoji_cluster']].drop_duplicates()

    tfidf = CountVectorizer(analyzer='char', lowercase=False)
    emoji_matrix = tfidf.fit_transform(df['emoji_str'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(emoji_matrix)
    df['emoji_cluster'] = clusters
    return df[['product_id', 'emoji_cluster']].drop_duplicates()

def lda_topic_features(df, num_topics=5, max_features=500):
    df = df.copy()
    df['review_text'] = df['review_text'].fillna("").astype(str)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=max_features)
    doc_term_matrix = vectorizer.fit_transform(df['review_text'])

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topic_matrix = lda.fit_transform(doc_term_matrix)

    topic_df = pd.DataFrame(topic_matrix, columns=[f"topic_{i}" for i in range(num_topics)])
    topic_df["product_id"] = df["product_id"].values
    topic_agg = topic_df.groupby("product_id").mean().reset_index()
    return topic_agg

def merge_features(topic_df, emoji_df):
    emoji_df['emoji_cluster'] = emoji_df['emoji_cluster'].astype(str)
    emoji_dummies = pd.get_dummies(emoji_df['emoji_cluster'], prefix='emoji_cluster')
    emoji_features = pd.concat([emoji_df[['product_id']], emoji_dummies], axis=1)
    merged = pd.merge(topic_df, emoji_features, on='product_id', how='left').fillna(0)
    return merged

def run_model(df, item_feature_df):
    dataset = Dataset()
    dataset.fit(df['user_id'], df['product_id'], item_features=item_feature_df.columns[1:].tolist())

    item_features_tuples = [
        (row['product_id'], [f for f, val in row[1:].items() if val > 0])
        for _, row in item_feature_df.iterrows()
    ]
    item_features = dataset.build_item_features(item_features_tuples)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    train_interactions, train_weights = dataset.build_interactions(
        [(row['user_id'], row['product_id'], row['rating']) for _, row in train_df.iterrows()]
    )
    test_interactions, _ = dataset.build_interactions(
        [(row['user_id'], row['product_id'], row['rating']) for _, row in test_df.iterrows()]
    )

    model = LightFM(loss='logistic', random_state=42)
    model.fit(train_interactions, sample_weight=train_weights, item_features=item_features, epochs=30, num_threads=1)

    metrics = {
        'AUC': auc_score(model, test_interactions, item_features=item_features).mean(),
        'Precision@5': precision_at_k(model, test_interactions, item_features=item_features, k=5).mean(),
        'Recall@5': recall_at_k(model, test_interactions, item_features=item_features, k=5).mean()
    }

    print("\n📊 LDA + Emoji Cluster Hybrid Model:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
def run_lda_pipeline(json_path: str):
    df = load_data(json_path, max_reviews=10000)
    topic_features = lda_topic_features(df, num_topics=5)
    emoji_features = cluster_emojis(df, n_clusters=5)
    combined_features = merge_features(topic_features, emoji_features)
    run_model(df, combined_features)


if __name__ == "__main__":
    filepath = "data/Musical_Instruments_5.json"
    df = load_data(filepath, max_reviews=10000)
    topic_features = lda_topic_features(df, num_topics=5)
    emoji_features = cluster_emojis(df, n_clusters=5)
    combined_features = merge_features(topic_features, emoji_features)
    run_model(df, combined_features)
