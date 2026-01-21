import argparse
import os
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.modeling.twotower_tfidf import TwoTowerTfidf
from src.modeling.dataset_tfidf_triplet import TfidfTripletDataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TripletCosineLoss(nn.Module):
    """
    Triplet loss using cosine similarity.
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, q_emb, p_emb, n_emb):
        # normalized already so between 1,0
        s_pos = (q_emb * p_emb).sum(dim=1)
        s_neg = (q_emb * n_emb).sum(dim=1)
        loss = torch.relu(self.margin - s_pos + s_neg).mean()
        return loss, s_pos.detach(), s_neg.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_triplets_csv", required=True)
    parser.add_argument(
        "--vectorizer_path",
        required=True,
        help="Path to fitted TF-IDF vectorizer (.joblib)"
    )
    parser.add_argument("--out_dir", default="artifacts/two_tower_tfidf_v1")

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TfidfTripletDataset(
        triplets_csv=args.train_triplets_csv,
        vectorizer_path=args.vectorizer_path,
    )



    tfidf_dim = train_ds.tfidf_dim 
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = TwoTowerTfidf(tfidf_dim=tfidf_dim, embed_dim=args.embed_dim, dropout=args.dropout).to(device)
    criterion = TripletCosineLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Save config
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for i, batch in enumerate(train_loader, start=1):
            q, p, n = batch
            q, p, n = q.to(device), p.to(device), n.to(device)

            optimizer.zero_grad(set_to_none=True)

            # turn into 3 embedcings
            q_emb = model.encode_resume(q)
            p_emb = model.encode_job(p)
            n_emb = model.encode_job(n)

            loss, s_pos, s_neg = criterion(q_emb, p_emb, n_emb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = running / 100
                running = 0.0
                print(
                    f"epoch {epoch} step {global_step} "
                    f"loss={avg_loss:.4f} "
                    f"pos_sim={s_pos.mean().item():.3f} neg_sim={s_neg.mean().item():.3f}"
                )

        # checkpoint each epoch
        ckpt_path = out_dir / f"model_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "tfidf_dim": tfidf_dim,
                "embed_dim": args.embed_dim,
            },
            ckpt_path,
        )
        print(f"saved checkpoint: {ckpt_path}")

    # final save
    final_path = out_dir / "model_final.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "tfidf_dim": tfidf_dim,
            "embed_dim": args.embed_dim,
        },
        final_path,
    )
    print(f"saved final model: {final_path}")


if __name__ == "__main__":
    main()


