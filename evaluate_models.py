import os
import matplotlib.pyplot as plt
import numpy as np
from main import load_movielens, split_train_test_timeaware, encode_train_matrix, \
                  predict_user_based, predict_item_based, predict_svd, precision_at_k

DATA_DIR = "data/ml-100k"
MODELS = ["user", "item", "svd"]
K_VALUES = [1, 3, 5, 10, 20]

def run_experiments():
    ratings, movies = load_movielens(DATA_DIR)
    train_df, test_df = split_train_test_timeaware(ratings, test_frac=0.2, seed=42)
    ed = encode_train_matrix(train_df)

    results = {}
    for model in MODELS:
        if model == "user":
            preds = predict_user_based(ed, k_neighbors=50)
        elif model == "item":
            preds = predict_item_based(ed, k_sim=50)
        elif model == "svd":
            preds = predict_svd(ed, n_components=50, random_state=42)

        precisions = []
        for K in K_VALUES:
            p_at_k, _ = precision_at_k(preds, ed, train_df, test_df, movies, K=K, min_relevant=4.0)
            precisions.append(p_at_k)
        results[model] = precisions
    return results

def plot_results(results, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)

    # Combined plot
    plt.figure(figsize=(6,4))
    for model, values in results.items():
        plt.plot(K_VALUES, values, marker="o", label=model)
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("Comparison of Recommendation Models")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "precision_at_k.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Individual plots
    for model, values in results.items():
        plt.figure(figsize=(6,4))
        plt.plot(K_VALUES, values, marker="o", label=model)
        plt.xlabel("K")
        plt.ylabel("Precision@K")
        plt.title(f"{model.upper()} Model - Precision@K")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(outdir, f"{model}.png"), dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    results = run_experiments()
    plot_results(results, outdir="plots")
