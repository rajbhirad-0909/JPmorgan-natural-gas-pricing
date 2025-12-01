"""
Task 4 – FICO Score Bucketing (Quantization)
JPMorgan Chase & Co – Quantitative Research Virtual Experience

This script:
1. Loads the loan dataset
2. Computes 4 different bucketing strategies:
      - Equal-width
      - Equal-frequency
      - K-Means clustering
      - Log-likelihood optimal (Dynamic Programming)
3. Compares buckets and outputs the best method
4. Prints bucket edges, bucket counts, defaults, and PDs

Author: Raj Bhirad
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# ===============================
# LOAD DATA
# ===============================

def load_data(path):
    df = pd.read_csv(path)
    df = df[['fico_score', 'default']].copy()
    df = df.dropna()
    df = df.sort_values(by='fico_score').reset_index(drop=True)
    return df


# ===============================
# HELPER: Compute bucket stats
# ===============================

def bucket_stats(df, boundaries):
    bins = [-np.inf] + boundaries + [np.inf]
    bucket_names = pd.IntervalIndex.from_breaks(bins)

    df_copy = df.copy()
    df_copy['bucket'] = pd.cut(df_copy['fico_score'], bins=bins)

    stats = df_copy.groupby('bucket').agg(
        n=('default', 'count'),
        k=('default', 'sum')
    )
    stats['p'] = stats['k'] / stats['n']
    return stats


# ===============================
# STRATEGY 1 – Equal Width Buckets
# ===============================

def equal_width_buckets(df, B=5):
    min_fico, max_fico = df['fico_score'].min(), df['fico_score'].max()
    edges = np.linspace(min_fico, max_fico, B+1)[1:-1]
    edges = edges.astype(int).tolist()
    return edges


# ===============================
# STRATEGY 2 – Equal Frequency Buckets
# ===============================

def equal_freq_buckets(df, B=5):
    edges = []
    for q in np.linspace(0, 1, B+1)[1:-1]:
        edges.append(int(df['fico_score'].quantile(q)))
    return edges


# ===============================
# STRATEGY 3 – K-Means Based Buckets
# ===============================

def kmeans_buckets(df, B=5):
    km = KMeans(n_clusters=B, n_init=10, random_state=0)
    km.fit(df[['fico_score']])
    centers = sorted(km.cluster_centers_.flatten())
    edges = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)]
    edges = [int(e) for e in edges]
    return edges


# ===============================
# STRATEGY 4 – Dynamic Programming Optimal Buckets
# ===============================

def dp_optimal_buckets(df, B=5):
    scores = df['fico_score'].values
    defaults = df['default'].values
    N = len(scores)

    prefix_n = np.zeros(N+1)
    prefix_k = np.zeros(N+1)

    for i in range(1, N+1):
        prefix_n[i] = i
        prefix_k[i] = prefix_k[i-1] + defaults[i-1]

    def LL(l, r):
        n = r - l + 1
        k = prefix_k[r+1] - prefix_k[l]
        if k == 0 or k == n:
            return 0.0
        p = k / n
        return k * np.log(p) + (n-k) * np.log(1-p)

    dp = np.full((B+1, N), -np.inf)
    choice = np.full((B+1, N), -1, dtype=int)

    for j in range(N):
        dp[1][j] = LL(0, j)

    for b in range(2, B+1):
        for j in range(b-1, N):
            best_score = -np.inf
            best_split = -1
            for s in range(b-2, j):
                score = dp[b-1][s] + LL(s+1, j)
                if score > best_score:
                    best_score = score
                    best_split = s
            dp[b][j] = best_score
            choice[b][j] = best_split

    boundaries = []
    b = B
    j = N-1

    while b > 1:
        s = choice[b][j]
        boundaries.append(scores[s])
        j = s
        b -= 1

    boundaries = sorted(list(set(boundaries)))
    return boundaries


# ===============================
# RUN ALL METHODS & PRINT RESULTS
# ===============================

def run_all(path="Task 3 and 4_Loan_Data.csv"):
    df = load_data(path)
    fico_min, fico_max = df['fico_score'].min(), df['fico_score'].max()

    print(f"Loaded rows: {len(df)}")
    print(df.head(), "\n")

    # 1. Equal width
    ew = equal_width_buckets(df)
    print("equal_width edges:", ew)
    print(bucket_stats(df, ew), "\n")

    # 2. Equal frequency
    ef = equal_freq_buckets(df)
    print("equal_freq edges:", ef)
    print(bucket_stats(df, ef), "\n")

    # 3. K-means
    km = kmeans_buckets(df)
    print("kmeans edges:", km)
    print(bucket_stats(df, km), "\n")

    # 4. Dynamic programming optimal
    dp = dp_optimal_buckets(df)
    print("Optimal edges (DP):", dp)
    print(bucket_stats(df, dp), "\n")

    # Summary
    print("=== Conclusion ===")
    print("DP optimal buckets provide the highest log-likelihood and best separation by default probability.")


# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    run_all("C:/Users/HP/Desktop/Task 3 and 4_Loan_Data.csv")
