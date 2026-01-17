import pandas as pd
import torch
from torch.utils.data import Dataset

class TripletTfidfDataset(Dataset):
    def __init__(self, csv_path: str, vectorizer):
        self.df = pd.read_csv(csv_path)

        r_text = self.df["resume_text"].astype(str).tolist()
        p_text = self.df["pos_job_text"].astype(str).tolist()
        n_text = self.df["neg_job_text"].astype(str).tolist()

        # Precompute
        self.R = vectorizer.transform(r_text)
        self.P = vectorizer.transform(p_text)
        self.N = vectorizer.transform(n_text)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, idx):
        r = torch.from_numpy(self.R[idx].toarray().ravel()).float()
        p = torch.from_numpy(self.P[idx].toarray().ravel()).float()
        n = torch.from_numpy(self.N[idx].toarray().ravel()).float()
        return r, p, n
