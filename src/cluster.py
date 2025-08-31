#!/usr/bin/env python3
"""
Mall Customers clustering pipeline:
- Loads CSV (expects data/Mall_Customers.csv or --data path)
- EDA scatterplots + histograms
- Scale features, compute elbow and silhouette scores to suggest k
- Fit KMeans (best k by silhouette) and visualize clusters
- Bonus: run DBSCAN and plot results
- Compute average spending per cluster and save summary CSV

Usage:
  python src/cluster.py --data data/Mall_Customers.csv
  (optional) --k 5  to force k
  (optional) --no-plots  to skip saving pngs
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# ---------- helper functions ----------

def load_data(path):
    """
    Load CSV and ensure expected columns present.
    """
    df = pd.read_csv(path)
    # Common Kaggle column names:
    expected = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    # If columns slightly different, attempt a forgiving match:
    for col in expected:
        if col not in df.columns:
            # try alt names
            if 'Annual' in col:
                # find a column that contains 'Income'
                matches = [c for c in df.columns if 'Income' in c]
                if matches:
                    df.rename(columns={matches[0]: 'Annual Income (k$)'}, inplace=True)
            if 'Spending' in col:
                matches = [c for c in df.columns if 'Spending' in c or 'Score' in c]
                if matches:
                    df.rename(columns={matches[0]: 'Spending Score (1-100)'}, inplace=True)
    # final check
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found columns: {df.columns.tolist()}")
    return df

def exploratory_plots(df, outdir='results', show=False):
    """
    Basic visual exploration:
    - scatter: Annual Income vs Spending Score (colored by Gender)
    - histograms for Income and Spending Score
    Saves plots to results/.
    """
    os.makedirs(outdir, exist_ok=True)

    # Scatter colored by Gender
    fig, ax = plt.subplots(figsize=(7,5))
    genders = df['Gender'].unique()
    colors = ['C0','C1','C2','C3']
    for i,g in enumerate(genders):
        subset = df[df['Gender']==g]
        ax.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                   label=g, alpha=0.7, s=40)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Income vs Spending Score (by Gender)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'scatter_income_spending_by_gender.png'))
    if show: plt.show()
    plt.close()

    # Histograms
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].hist(df['Annual Income (k$)'], bins=12)
    axs[0].set_title('Annual Income (k$)')
    axs[1].hist(df['Spending Score (1-100)'], bins=12)
    axs[1].set_title('Spending Score (1-100)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'hist_income_spending.png'))
    if show: plt.show()
    plt.close()

def scale_features(df, features=['Annual Income (k$)','Spending Score (1-100)']):
    """
    Scale selected features with StandardScaler. Return scaled array and scaler.
    Scaling is important for distance-based algorithms (KMeans, DBSCAN).
    """
    X = df[features].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def compute_elbow(X, kmax=10, outpath='results/elbow.png'):
    """
    Compute and plot inertia (sum of squared distances) for k=1..kmax
    Elbow helps find when inertia improvement diminishes.
    """
    inertias = []
    Ks = list(range(1, kmax+1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(Ks, inertias, '-o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow method (inertia vs k)')
    plt.xticks(Ks)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return Ks, inertias

def compute_silhouette(X, kmin=2, kmax=10, outpath='results/silhouette.png'):
    """
    Compute silhouette score for k in [kmin,kmax]. Higher silhouette -> better separated clusters.
    Returns dict {k:score}.
    """
    scores = {}
    Ks = list(range(kmin, kmax+1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
    # plot
    plt.figure(figsize=(6,4))
    plt.plot(list(scores.keys()), list(scores.values()), '-o')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette scores for different k')
    plt.xticks(Ks)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return scores

def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    return km, labels

def plot_kmeans_results(df, X_scaled, labels, km_model, scaler, outdir='results'):
    """
    Scatter plot in original feature space with cluster colors and centroids.
    We convert centroids back to original space using inverse_transform(scaler).
    """
    os.makedirs(outdir, exist_ok=True)
    centers_scaled = km_model.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)

    plt.figure(figsize=(7,5))
    scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                          c=labels, cmap='tab10', s=40, alpha=0.8)
    plt.scatter(centers_orig[:,0], centers_orig[:,1], marker='X', s=200, c='black', label='centroids')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title(f'KMeans clusters (k={km_model.n_clusters})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'kmeans_k{km_model.n_clusters}_clusters.png'))
    plt.close()

def run_dbscan(X, df, scaler, outdir='results', eps=None, min_samples=5):
    """
    Run DBSCAN on scaled features. We provide a k-distance plot to help choose eps.
    If eps is None, default 0.5 is used (after scaling).
    """
    os.makedirs(outdir, exist_ok=True)
    # k-distance plot
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    # k-distance is the distance to the 4-th neighbor (index 4)
    kdist = np.sort(distances[:,4])
    plt.figure(figsize=(6,4))
    plt.plot(kdist)
    plt.ylabel('4-NN distance (sorted)')
    plt.xlabel('Samples sorted by 4-NN distance')
    plt.title('k-distance graph (k=5) — look for knee to choose eps')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'dbscan_kdistance.png'))
    plt.close()

    if eps is None:
        eps = 0.5  # default; user can change after inspecting k-distance plot

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = db.fit_predict(X)
    # number of clusters (ignoring noise label -1)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    # plot dbscan result in original space
    plt.figure(figsize=(7,5))
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                c=db_labels, cmap='tab10', s=40, alpha=0.8)
    plt.title(f'DBSCAN (eps={eps}, min_samples={min_samples}) => clusters={n_clusters}')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'dbscan_eps{eps}_min{min_samples}.png'))
    plt.close()

    return db, db_labels

def summarize_clusters(df, labels, outpath='results/cluster_summary.csv'):
    """
    Add column 'Cluster' and compute aggregates: count, mean income, mean spending.
    Save a CSV summary.
    """
    df2 = df.copy()
    df2['Cluster'] = labels
    summary = df2.groupby('Cluster').agg(
        Count=('CustomerID','count'),
        Mean_Income_k=('Annual Income (k$)','mean'),
        Mean_Spending_Score=('Spending Score (1-100)','mean')
    ).reset_index().sort_values('Cluster')
    summary.to_csv(outpath, index=False)
    return summary, df2

# ---------- main pipeline ----------

def main(args):
    print("Loading data...", args.data)
    df = load_data(args.data)

    print("Running exploratory plots...")
    exploratory_plots(df, outdir=args.results)

    print("Scaling features...")
    Xs, scaler = scale_features(df)

    print("Computing elbow plot...")
    compute_elbow(Xs, kmax=10, outpath=os.path.join(args.results,'elbow.png'))

    print("Computing silhouette scores...")
    sil_scores = compute_silhouette(Xs, kmin=2, kmax=10,
                                    outpath=os.path.join(args.results,'silhouette.png'))

    if args.k is not None:
        k_choice = args.k
        print(f"User-specified k={k_choice}")
    else:
        # choose k with highest silhouette score (simple automated choice)
        k_choice = max(sil_scores, key=sil_scores.get)
        print(f"Auto-selected k (max silhouette) = {k_choice}")

    print(f"Fitting KMeans with k={k_choice} ...")
    km, labels = fit_kmeans(Xs, k_choice)

    print("Plotting KMeans clusters...")
    plot_kmeans_results(df, Xs, labels, km, scaler, outdir=args.results)

    print("Summarizing clusters (averages)...")
    summary, df_with_clusters = summarize_clusters(df, labels,
                                                   outpath=os.path.join(args.results,'cluster_summary.csv'))
    print(summary.to_string(index=False))

    # Bonus: DBSCAN
    if args.run_dbscan:
        print("Running DBSCAN (bonus). Inspect results/dbscan_kdistance.png to choose eps if desired.")
        db, db_labels = run_dbscan(Xs, df, scaler, outdir=args.results,
                                   eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        print(f"DBSCAN produced {n_clusters_db} clusters (noise = label -1).")

    print("Done. Results saved in:", args.results)
    print("Cluster summary CSV at:", os.path.join(args.results,'cluster_summary.csv'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/Mall_Customers.csv', help='Path to CSV')
    p.add_argument('--k', type=int, default=None, help='Force k for KMeans')
    p.add_argument('--results', default='results', help='Output folder for plots and CSVs')
    p.add_argument('--no-plots', action='store_true', help='(not used) skip plots')
    p.add_argument('--run-dbscan', action='store_true', help='Run DBSCAN as bonus')
    p.add_argument('--dbscan-eps', type=float, default=None, help='eps for DBSCAN (scaled space)')
    p.add_argument('--dbscan-min-samples', type=int, default=5, help='min_samples for DBSCAN')
    args = p.parse_args()
    main(args)
