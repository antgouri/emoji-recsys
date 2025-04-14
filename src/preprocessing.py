#The preprocessing - basic level
import pandas as pd
import json

def load_amazon_reviews(file_path: str, max_reviews: int = None) -> pd.DataFrame:
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_reviews and i >= max_reviews:
                break
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    df = df.rename(columns={
        'reviewerID': 'user_id',
        'asin': 'product_id',
        'overall': 'rating',
        'reviewText': 'review_text'
    })
    
    df = df[['user_id', 'product_id', 'rating', 'review_text']].dropna()
    return df
