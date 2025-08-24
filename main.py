# main.py
import argparse
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------------------------
# Utils & Data Loading
# ---------------------------

def load_movielens(data_dir: str):
    """
    Loads ratings and movie titles from MovieLens 100k in either legacy (u.data/u.item)
    or Kaggle-style (ratings.csv/movies.csv) file names.
    Returns: ratings_df [userId, movieId, rating, timestamp], movies_df [movieId, title]
    """
    data_dir = os.path.abspath(data_dir)
    # Try Kaggle-style first
    ratings_path_csv = os.path.join(data_dir, "ratings.csv")
    movies_path_csv = os.path.join(data_dir, "movies.csv")

    # Legacy GroupLens style
    ratings_path_u = os.path.join(data_dir, "u.data")
    movies_path_u = os.path.join(data_dir, "u.item")

    if os.path.exists(ratings_path_csv):
        ratings = pd.read_csv(ratings_path_csv)
        # ensure columns exist
        expected_cols = {"userId", "movieId", "rating", "timestamp"}
        missing = expected_cols - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings.csv missing columns: {missing}")
        if os.path.exists(movies_path_csv):
            movies = pd.read_csv(movies_path_csv)
            if "movieId" not in movies.columns or "title" not in movies.columns:
                raise ValueError("movies.csv must have columns: movieId,title")
        else:
            movies = pd.DataFrame({"movieId": ratings["movieId"].unique(), "title": [None]*ratings["movieId"].nunique()})
    elif os.path.exists(ratings_path_u):
        # u.data is tab-separated with no header: user item rating timestamp
        ratings = pd.read_csv(ratings_path_u, sep="\t", names=["userId", "movieId", "rating", "timestamp"])
        # u.item is pipe-separated; first two columns are movieId|title|...
        if os.path.exists(movies_path_u):
            movies = pd.read_csv(movies_path_u, sep="|", header=None, encoding="latin-1")
            movies = movies[[0, 1]].rename(columns={0: "movieId", 1: "title"})
        else:
            movies = pd.DataFrame({"movieId": ratings["movieId"].unique(), "title": [None]*ratings["movieId"].nunique()})
    else:
        raise FileNotFoundError(
            f"Couldn't find ratings file. Put ratings.csv (Kaggle) or u.data (GroupLens) in {data_dir}"
        )

    # enforce dtypes
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    if "timestamp" not in ratings.columns:
        ratings["timestamp"] = 0

    movies = movies.drop_duplicates(subset=["movieId"])
    return ratings, movies

@dataclass
class EncodedData:
    R_train: sp.csr_matrix     # users x items (train)
    user_ids: np.ndarray       # index -> userId
    item_ids: np.ndarray       # index -> movieId
    user_index: dict           # userId -> index
    item_index: dict           # movieId -> index
    user_means: np.ndarray     # mean rating per user on train

def encode_train_matrix(train_df: pd.DataFrame):
    users = np.sort(train_df["userId"].unique())
    items = np.sort(train_df["movieId"].unique())
    u_index = {u:i for i,u in enumerate(users)}
    i_index = {m:i for i,m in enumerate(items)}

    row = train_df["userId"].map(u_index).values
    col = train_df["movieId"].map(i_index).values
    data = train_df["rating"].values

    R = sp.csr_matrix((data, (row, col)), shape=(len(users), len(items)))

    # user means on non-zero entries
    user_means = np.zeros(R.shape[0], dtype=float)
    for u in range(R.shape[0]):
        start, end = R.indptr[u], R.indptr[u+1]
        if end > start:
            user_means[u] = R.data[start:end].mean()
        else:
            user_means[u] = 0.0

    return EncodedData(R, users, items, u_index, i_index, user_means)

def split_train_test_timeaware(ratings: pd.DataFrame, test_frac=0.2, seed=42):
    """
    For each user: sort by timestamp and put ~last 20% into test (at least 1).
    """
    rng = np.random.default_rng(seed)
    train_rows = []
    test_rows = []
    for uid, df_u in ratings.groupby("userId"):
        df_u = df_u.sort_values("timestamp")
        n = len(df_u)
        n_test = max(1, int(np.ceil(n * test_frac)))
        # choose last n_test as test (time-aware)
        test_u = df_u.tail(n_test)
        train_u = df_u.iloc[: n - n_test]
        train_rows.append(train_u)
        test_rows.append(test_u)
    train = pd.concat(train_rows, ignore_index=True)
    test  = pd.concat(test_rows,  ignore_index=True)
    return train, test

# ---------------------------
# Models
# ---------------------------

def predict_user_based(ed: EncodedData, k_neighbors=50):
    """
    User-based CF with mean-centering and cosine similarity.
    Returns dense predictions (users x items).
    """
    R = ed.R_train.tocsr()
    # mean-center (sparse-aware)
    Rc = R.copy().astype(float)
    for u in range(Rc.shape[0]):
        start, end = Rc.indptr[u], Rc.indptr[u+1]
        if end > start:
            Rc.data[start:end] -= ed.user_means[u]

    # dense views (ml-100k sizes are OK)
    Rc_dense = Rc.toarray()
    R_mask = (R.toarray() > 0).astype(float)

    # user-user cosine
    S = cosine_similarity(Rc_dense)  # (U x U)
    np.fill_diagonal(S, 0.0)

    # Optionally keep only top-k neighbors per user
    if k_neighbors is not None and k_neighbors > 0 and k_neighbors < S.shape[1]:
        # sparsify by zeroing out everything except top-k per row
        idx = np.argpartition(-S, kth=k_neighbors, axis=1)[:, :k_neighbors]
        S_pruned = np.zeros_like(S)
        rows = np.arange(S.shape[0])[:, None]
        S_pruned[rows, idx] = S[rows, idx]
        S = S_pruned

    # Numerator & denominator for each item
    numer = S @ Rc_dense                         # (U x I)
    denom = np.maximum(1e-8, (np.abs(S) @ R_mask))  # normalize by avail ratings

    pred_centered = numer / denom
    preds = pred_centered + ed.user_means[:, None]
    return preds

def predict_item_based(ed: EncodedData, k_sim=50):
    """
    Item-based CF using item-item cosine. Returns dense predictions.
    """
    R = ed.R_train.tocsr()
    R_dense = R.toarray()
    R_mask = (R_dense > 0).astype(float)

    # item-item cosine on columns
    # cosine_similarity expects rows = samples, so pass R.T
    W = cosine_similarity(R_dense.T)  # (I x I)
    np.fill_diagonal(W, 0.0)

    if k_sim is not None and 0 < k_sim < W.shape[1]:
        idx = np.argpartition(-W, kth=k_sim, axis=1)[:, :k_sim]
        W_pruned = np.zeros_like(W)
        rows = np.arange(W.shape[0])[:, None]
        W_pruned[rows, idx] = W[rows, idx]
        W = W_pruned

    numer = R_dense @ W                 # (U x I)
    denom = np.maximum(1e-8, R_mask @ np.abs(W))
    preds = numer / denom
    return preds

def predict_svd(ed: EncodedData, n_components=20, random_state=42):
    """
    Matrix factorization via TruncatedSVD on a mean-centered matrix.
    Reconstructs low-rank approximation and adds user means back.
    """
    R = ed.R_train.tocsr()
    Rc = R.copy().astype(float)
    for u in range(Rc.shape[0]):
        start, end = Rc.indptr[u], Rc.indptr[u+1]
        if end > start:
            Rc.data[start:end] -= ed.user_means[u]
    Rc_dense = Rc.toarray()

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    U_s = svd.fit_transform(Rc_dense)       # (U x k) == (U*S)
    Vt = svd.components_                    # (k x I)
    Rc_hat = U_s @ Vt                       # (U x I) approx of centered
    preds = Rc_hat + ed.user_means[:, None]
    return preds

# ---------------------------
# Evaluation
# ---------------------------

def precision_at_k(preds, ed: EncodedData, train_df: pd.DataFrame, test_df: pd.DataFrame, movies_df: pd.DataFrame,
                   K=10, min_relevant=4.0):
    """
    Precision@K: for each user, count how many of their *relevant* (>= min_relevant) test items
    appear in the top-K recommendations (excluding items seen in train).
    """
    # Build quick lookup of train items per user (exclude from recommendation)
    train_items_by_user = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    test_rel_by_user = test_df[test_df["rating"] >= min_relevant].groupby("userId")["movieId"].apply(set).to_dict()

    user_precisions = []
    all_users = ed.user_ids
    for uidx, uid in enumerate(all_users):
        relevant = test_rel_by_user.get(uid, set())
        if len(relevant) == 0:
            continue  # skip users with no relevant items in test

        scores = preds[uidx].copy()
        seen = train_items_by_user.get(uid, set())

        # Mask seen items
        mask = np.ones(scores.shape[0], dtype=bool)
        if seen:
            seen_idx = [ed.item_index[m] for m in seen if m in ed.item_index]
            mask[seen_idx] = False

        # get top-K indices among unseen
        unseen_scores = scores[mask]
        if unseen_scores.size == 0:
            user_precisions.append(0.0)
            continue

        unseen_indices = np.where(mask)[0]
        topk_unseen_relpos = np.argpartition(-unseen_scores, kth=min(K, unseen_scores.size)-1)[:K]
        topk_item_idx = unseen_indices[topk_unseen_relpos]
        topk_movie_ids = set(ed.item_ids[topk_item_idx])

        hits = len(relevant & topk_movie_ids)
        user_precisions.append(hits / float(K))

    if len(user_precisions) == 0:
        return 0.0, 0
    return float(np.mean(user_precisions)), len(user_precisions)

# ---------------------------
# Driver
# ---------------------------

def run(data_dir, model, K, min_relevant, test_frac, k_neighbors, k_sim, svd_components, seed):
    ratings, movies = load_movielens(data_dir)
    # split
    train_df, test_df = split_train_test_timeaware(ratings, test_frac=test_frac, seed=seed)

    # encode on TRAIN (only items in train are recommendable by this simple setup)
    ed = encode_train_matrix(train_df)

    # choose model
    if model == "user":
        preds = predict_user_based(ed, k_neighbors=k_neighbors)
    elif model == "item":
        preds = predict_item_based(ed, k_sim=k_sim)
    elif model == "svd":
        preds = predict_svd(ed, n_components=svd_components, random_state=seed)
    else:
        raise ValueError("model must be one of: user | item | svd")

    p_at_k, n_users = precision_at_k(preds, ed, train_df, test_df, movies, K=K, min_relevant=min_relevant)

    print(f"\nModel: {model}")
    print(f"Users evaluated: {n_users}")
    print(f"Precision@{K} (rating >= {min_relevant} as relevant): {p_at_k:.4f}")

    # show a tiny demo for a random user
    demo_uid = int(ed.user_ids[np.random.default_rng(seed).integers(low=0, high=len(ed.user_ids))])
    recs = recommend_for_user(demo_uid, preds, ed, movies, train_df, topn=10)
    print(f"\nSample recommendations for user {demo_uid}:")
    for rank, (mid, title, score) in enumerate(recs, start=1):
        print(f"{rank:2d}. [{mid}] {title}  (score={score:.3f})")

def recommend_for_user(user_id, preds, ed: EncodedData, movies_df: pd.DataFrame, train_df: pd.DataFrame, topn=10):
    if user_id not in ed.user_index:
        raise ValueError("User not in training set.")
    uidx = ed.user_index[user_id]
    scores = preds[uidx].copy()

    seen = set(train_df.loc[train_df["userId"] == user_id, "movieId"].tolist())
    # Mask seen
    mask = np.ones(scores.shape[0], dtype=bool)
    if seen:
        seen_idx = [ed.item_index[m] for m in seen if m in ed.item_index]
        mask[seen_idx] = False
    unseen_scores = scores[mask]
    unseen_indices = np.where(mask)[0]

    topn_relpos = np.argpartition(-unseen_scores, kth=min(topn, unseen_scores.size)-1)[:topn]
    top_item_idx = unseen_indices[topn_relpos]
    top_scores = unseen_scores[topn_relpos]
    order = np.argsort(-top_scores)
    top_item_idx = top_item_idx[order]
    top_scores = top_scores[order]

    mid_list = ed.item_ids[top_item_idx]
    titles = movies_df.set_index("movieId").reindex(mid_list)["title"].fillna("").values
    out = [(int(m), str(t), float(s)) for m, t, s in zip(mid_list, titles, top_scores)]
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommendation System (User/Item CF + SVD)")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to MovieLens data folder")
    parser.add_argument("--model", type=str, default="user", choices=["user", "item", "svd"])
    parser.add_argument("--K", type=int, default=10, help="K for Precision@K and top-N")
    parser.add_argument("--min_relevant", type=float, default=4.0, help="Rating threshold to consider relevant")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Per-user test fraction (time-aware split)")
    parser.add_argument("--k_neighbors", type=int, default=50, help="k nearest neighbors for user-based")
    parser.add_argument("--k_sim", type=int, default=50, help="k most similar items for item-based")
    parser.add_argument("--svd_components", type=int, default=20, help="latent factors for SVD")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        model=args.model,
        K=args.K,
        min_relevant=args.min_relevant,
        test_frac=args.test_frac,
        k_neighbors=args.k_neighbors,
        k_sim=args.k_sim,
        svd_components=args.svd_components,
        seed=args.seed
    )
