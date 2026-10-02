"""recommender.py - loads everything ONCE at startup and serves recommendations for the Flask app."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv

products   = pd.read_csv("products_expanded.csv")            # rows in product_id order 1..200
browsing   = pd.read_csv("browsing_history_expanded.csv")
purchases  = pd.read_csv("purchases_expanded.csv")
n_users    = len(pd.read_csv("users_expanded.csv"))
n_products = len(products)
image_emb  = np.load("image_emb.npy")                        # (200, 2048) ResNet50, from notebook 05
text_emb   = np.load("text_emb.npy")                         # (200, 384) MiniLM, from notebook 05

BROWSE_W, BUY_W = 1.0, 4.0                                   # same weights as in evaluation
b = browsing[["user_id", "product_id"]].assign(weight=BROWSE_W)
p = purchases[["user_id", "product_id"]].assign(weight=BUY_W)
inter = pd.concat([b, p]).groupby(["user_id", "product_id"], as_index=False)["weight"].sum()
R = csr_matrix((inter["weight"].values, (inter["user_id"].values - 1, inter["product_id"].values - 1)),
               shape=(n_users, n_products))
item_sim = cosine_similarity(R.T); np.fill_diagonal(item_sim, 0)
seen = inter.groupby("user_id")["product_id"].apply(set).to_dict()   # products each user already touched


def item_cf_scores(uid):
    """Products similar (by co-interaction) to what this user touched."""
    return R[uid - 1].toarray().ravel() @ item_sim


def content_scores(uid):
    """Cosine similarity between each product's text embedding and the user's taste vector."""
    mine = inter[inter["user_id"] == uid]
    w = mine["weight"].values[:, None]
    profile = (w * text_emb[mine["product_id"].values - 1]).sum(axis=0) / w.sum()
    return text_emb @ (profile / np.linalg.norm(profile))


def minmax(x):
    spread = x.max() - x.min()
    return (x - x.min()) / spread if spread > 0 else x * 0


def hybrid_scores(uid, alpha=0.75):
    """Final model: 75% item-CF + 25% content, each min-max scaled over UNSEEN products only."""
    unseen = ~np.isin(np.arange(1, n_products + 1), list(seen.get(uid, ())))
    out = np.full(n_products, -np.inf)                           # seen products get -infinity
    out[unseen] = alpha * minmax(item_cf_scores(uid)[unseen]) + (1 - alpha) * minmax(content_scores(uid)[unseen])
    return out


class MultiModalRec(nn.Module):          # SAME architecture as notebook 07 (needed to load the weights)
    def __init__(self, n_users, n_products, dim=64, use_gcn=True):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_products, dim)
        self.img_proj = nn.Linear(2048, dim)
        self.txt_proj = nn.Linear(384, dim)
        self.gcn = GCNConv(dim, dim) if use_gcn else None

    def product_vectors(self, img, txt, edge_index):
        x = self.item_emb.weight + self.img_proj(img) + self.txt_proj(txt)
        return x + self.gcn(x, edge_index) if self.gcn is not None else x


mm_model = MultiModalRec(n_users, n_products)
mm_model.load_state_dict(torch.load("multimodal_gcn.pt"))      # the weights trained in notebook 07
mm_model.eval()                                                # inference mode
with torch.no_grad():
    prod = mm_model.product_vectors(torch.tensor(image_emb, dtype=torch.float32),
                                    torch.tensor(text_emb, dtype=torch.float32),
                                    torch.tensor(np.load("edge_index.npy"), dtype=torch.long))
    MM_SCORES = (mm_model.user_emb.weight @ prod.T).numpy()   # (500, 200): every user x product, once

pop_scores = purchases["product_id"].value_counts().reindex(products["product_id"], fill_value=0).values.astype(float)

METHODS = {"hybrid": hybrid_scores, "item_cf": item_cf_scores, "content": content_scores,
           "multimodal": lambda uid: MM_SCORES[uid - 1], "popular": lambda uid: pop_scores}


def recommend(uid, method="hybrid", k=10):
    """Top-k unseen products for a user. Unknown users fall back to 'popular' (cold start)."""
    if uid not in seen:
        method = "popular"
    recs = products.copy()
    recs["score"] = METHODS[method](uid)
    recs = recs[~recs["product_id"].isin(seen.get(uid, set()))]     # never re-recommend seen items
    return recs.sort_values("score", ascending=False).head(k), method
