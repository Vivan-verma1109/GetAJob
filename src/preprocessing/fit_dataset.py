import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from dataset import TripletTfidfDataset

train_path = "data/processed/triplets/train_triplets.csv"
dev_path   = "data/processed/triplets/dev_triplets.csv"

df_train = pd.read_csv(train_path)

corpus = pd.concat([
    df_train["resume_text"].astype(str),
    df_train["pos_job_text"].astype(str),
    df_train["neg_job_text"].astype(str),
], axis=0).tolist()

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
vectorizer.fit(corpus)

train_ds = TripletTfidfDataset(train_path, vectorizer)
dev_ds   = TripletTfidfDataset(dev_path, vectorizer)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
dev_loader   = DataLoader(dev_ds, batch_size=64, shuffle=False, num_workers=0)
