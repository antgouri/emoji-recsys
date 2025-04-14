
import pandas as pd
from textblob import TextBlob
import os

def get_sentiment_emoji(text: str) -> str:
    if not isinstance(text, str):
        return ''
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.3:
        return " 😊"
    elif sentiment < -0.3:
        return " 😞"
    else:
        return " 😐"
        
def compute_item_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    from textblob import TextBlob

    df = df.copy()
    df["sentiment_score"] = df["review_text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Aggregate average sentiment score per item
    item_sentiments = df.groupby("product_id")["sentiment_score"].mean().reset_index()
    item_sentiments.columns = ["product_id", "avg_sentiment"]
    
    return item_sentiments


def inject_emojis_to_reviews(df: pd.DataFrame, review_col: str = "review_text") -> pd.DataFrame:
    df = df.copy()
    df["emoji_review"] = df[review_col].apply(lambda x: str(x) + get_sentiment_emoji(x))
    return df

def save_emoji_reviews(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
