#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:21:42 2026
@author: maximilianschulten
"""
#%% Setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.gen_utils import load_embedder
from lib.scoring import embed_doc
import numpy as np
import pandas as pd

out_dir = os.path.join(os.path.dirname(__file__), "dist_data")

#%% Data
all_resumes = pd.read_csv(os.path.join(os.path.dirname(__file__), "..") + "/data/resume-class.csv")
jds = pd.read_csv(os.path.join(os.path.dirname(__file__), "..") + "/data/job-descriptions.csv").sample(n=200,
                                                                                                        ignore_index=True,
                                                                                                        random_state=42)
n_per_cat = 3
resumes = pd.concat([
    g.sample(n=min(len(g), n_per_cat), random_state=42)
    for _, g in all_resumes.groupby("Category")
]).reset_index(drop=True)
print(f"Stratified resume sample: {len(resumes)} resumes across {resumes['Category'].nunique()} categories")

#%% Embed + collect best-per-window similarities
emb = load_embedder()

best_per_window = []   # every best_per_window value across all pairs
pair_positions      = []   # normalized position of each window [0,1]
pair_scores         = []   # the similarity value
pair_categories     = []   # resume category for that pair

vars_ = ['best_per_window', 'pair_positions', 'pair_scores', 'pair_categories']

n_cache_hits = 0

for var in vars_:
    dir_ = os.path.join(out_dir, f"{var}.npy")
    print(dir_)
    if not os.path.exists(dir_):
        print("Miss")
        continue
    print("hit")
    data = np.load(dir_)
    exec(f"{var}=data")
    n_cache_hits += 1
    
cached = n_cache_hits == len(vars_)

if not cached:
    for idx, jdrow in jds.iterrows():
        jd_emb, _ = embed_doc(jdrow['job_description'], embedding_model=emb, window_size=30, stride=10)
        n_job_windows = len(jd_emb)
        positions = np.linspace(0, 1, n_job_windows)
    
        for jdx, resrow in resumes.iterrows():
            if jdx % 10 == 0:
                print("#", idx, '.', jdx, sep="")
            res_emb, _ = embed_doc(resrow['Text'], embedding_model=emb, window_size=30, stride=10)
    
            sim_mat = jd_emb @ res_emb.T
            best_per_window_ = sim_mat.max(axis=1)
    
            best_per_window.extend(best_per_window_.tolist())
            pair_positions.extend(positions.tolist())
            pair_scores.extend(best_per_window_.tolist())
            pair_categories.extend([resrow['Category']] * n_job_windows)
    
    best_per_window = np.array(best_per_window)
    pair_positions      = np.array(pair_positions)
    pair_scores         = np.array(pair_scores)

print(f"Total window similarities collected: {len(best_per_window):,}")

#%% Save
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "best_per_window.npy"),  best_per_window)
np.save(os.path.join(out_dir, "pair_positions.npy"),   pair_positions)
np.save(os.path.join(out_dir, "pair_scores.npy"),      pair_scores)
np.save(os.path.join(out_dir, "pair_categories.npy"),  np.array(pair_categories))
print(f"Saved to {out_dir}")

#%% Global distribution of best-per-window similarities
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, beta

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Raw histogram + KDE
axes[0].hist(best_per_window, bins=60, density=True, alpha=0.6, edgecolor='black')
kde = gaussian_kde(best_per_window)
xs = np.linspace(best_per_window.min(), best_per_window.max(), 300)
axes[0].plot(xs, kde(xs), linewidth=2, label='KDE')
mu_fit, std_fit = norm.fit(best_per_window)
alpha_fit, beta_fit, loc, scale = beta.fit(best_per_window)
axes[0].plot(xs, norm.pdf(xs, mu_fit, std_fit), linewidth=2, linestyle='--', label=f'Normal fit (μ={mu_fit:.3f}, σ={std_fit:.3f})')
axes[0].plot(xs, beta.pdf(xs, alpha_fit, beta_fit, loc, scale), linewidth=2, linestyle='--', label=f'Beta fit')
axes[0].set_title('Best-per-window similarity distribution')
axes[0].set_xlabel('Cosine similarity')
axes[0].set_ylabel('Density')
axes[0].legend()

# Mean similarity by position bucket — shows where signal lives in the JD
n_buckets = 20
bucket_means = []
bucket_stds  = []
bucket_centers = []
for i in range(n_buckets):
    lo, hi = i / n_buckets, (i + 1) / n_buckets
    mask = (pair_positions >= lo) & (pair_positions < hi)
    if mask.sum() > 0:
        bucket_means.append(pair_scores[mask].mean())
        bucket_stds.append(pair_scores[mask].std())
        bucket_centers.append((lo + hi) / 2)

bucket_means = np.array(bucket_means)
bucket_stds  = np.array(bucket_stds)
bucket_centers = np.array(bucket_centers)

axes[1].plot(bucket_centers, bucket_means, linewidth=2, label='Mean similarity')
axes[1].fill_between(bucket_centers,
                     bucket_means - bucket_stds,
                     bucket_means + bucket_stds,
                     alpha=0.2, label='±1 std')
axes[1].set_title('Mean best-per-window similarity by JD position')
axes[1].set_xlabel('Normalized JD position')
axes[1].set_ylabel('Mean cosine similarity')
axes[1].legend()

plt.tight_layout()
plt.show()

#%% Fit normal to similarity distribution
print(f"Normal fit: mu={mu_fit:.4f}, std={std_fit:.4f}")
from scipy.stats import skew, kurtosis, shapiro
print(f"Skewness: {skew(best_per_window):.4f}")
print(f"Kurtosis: {kurtosis(best_per_window):.4f}")