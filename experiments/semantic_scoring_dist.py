#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:24:16 2026
@author: maximilianschulten
"""
#%% Setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.gen_utils import load_embedder
from lib.scoring import semantic_score, embed_doc
import numpy as np
import pandas as pd
from scipy.stats import shapiro

out_dir = os.path.join(os.path.dirname(__file__), "dist_data")
#%% Data
all_resumes = pd.read_csv(os.path.join(os.path.dirname(__file__), "..") + "/data/resume-class.csv")
jds = pd.read_csv(os.path.join(os.path.dirname(__file__), "..") + "/data/job-descriptions.csv").sample(n=200, 
                                                                                                       ignore_index=True, 
                                                                                                      random_state=42)
# Stratified sample — equal representation per category, up to n_per_cat each
n_per_cat = 3
resumes = pd.concat([
    g.sample(n=min(len(g), n_per_cat), random_state=42)
    for _, g in all_resumes.groupby("Category")
]).reset_index(drop=True)
print(f"Stratified resume sample: {len(resumes)} resumes across {resumes['Category'].nunique()} categories")
print(resumes['Category'].value_counts())

#%% Embed
emb = load_embedder()
sem_scores = []
categories = []

vars_ = ['sem_scores', 'categories']

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
        for jdx, resrow in resumes.iterrows():
            if jdx % 10 == 0:
                print("#", idx, '.', jdx, sep="")
            res_emb, _ = embed_doc(resrow['Text'], embedding_model=emb, window_size=30, stride=10)
            score_ = semantic_score(res_emb, jd_emb, mu=0.3, sigma=0.2)
            sem_scores.append(score_)
            categories.append(resrow['Category'])

results = pd.DataFrame({
    'category': categories,
    'semantic_score': sem_scores,
})

#%% Distribution approximation
import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(15, 4))
axes.hist(sem_scores, bins=30, edgecolor='black')
axes.set_title('Semantic Scores')
axes.set_xlabel('Score')
axes.set_ylabel('Count')
plt.tight_layout()
plt.show()

#%% KDE
from scipy.stats import gaussian_kde, norm, beta
fig, axes = plt.subplots(figsize=(7, 4))
data_list = [
    (sem_scores, 'Distribution of Semantic Scores')
]
for ax, (data, title) in zip([axes], data_list):
    data = np.array(data)
    nonzero = data[data != 0]
    ax.hist(nonzero, bins=50, edgecolor='black', density=True, alpha=0.6)
    kde = gaussian_kde(nonzero)
    a, b, loc, scale = beta.fit(sem_scores)
    m, s = norm.fit(sem_scores)
    xs = np.linspace(nonzero.min(), nonzero.max(), 300)
    ax.plot(xs, kde(xs), linewidth=2, label="Gaussian KDE Estimate", linestyle='--', c='maroon')
    ax.plot(xs, beta.pdf(xs, a, b, loc, scale), linewidth=2, label=rf'$\alpha={a:.2f}$, $\beta = {b:.2f}$', linestyle='-.', c='red')
    ax.plot(xs, norm.pdf(xs, m,s), linewidth=2, label=rf'$\mu={m:.2f}$, $\sigma = {s:.2f}$', linestyle=':', c='coral')
    ax.set_title(title)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

#%% Save
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "sem_scores.npy"), np.array(sem_scores))
np.save(os.path.join(out_dir, "categories.npy"), np.array(categories))

results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
print(f"Saved {len(sem_scores)} records to {out_dir}")

#%% Per-category distribution
fig, ax = plt.subplots(figsize=(12, 5))
for cat, group in results.groupby('category'):
    scores = group['semantic_score'].values
    if len(scores) >= 20:
        mu, std = norm.fit(scores)
        xs = np.linspace(scores.min(), scores.max(), 200)
        ax.plot(xs, norm.pdf(xs, mu, std), label=f"{cat} (μ={mu:.2f})")
ax.set_title('Per-category semantic score distributions')
ax.set_xlabel('Semantic score')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

#%% Fit global rescaling params
global_mu, global_std = norm.fit(sem_scores)
print(f"Global fit: mu={global_mu:.4f}, std={global_std:.4f}")

category_params = {}
for cat, group in results.groupby('category'):
    scores = group['semantic_score'].values
    if len(scores) >= 30:
        mu, std = norm.fit(scores)
        category_params[cat] = (mu, std)
        print(f"{cat}: mu={mu:.4f}, std={std:.4f}")
        
#%%
print(shapiro(sem_scores))
from scipy.stats import skew, kurtosis
print(f"Skewness: {skew(sem_scores):.4f}")   # ~0 for normal
print(f"Kurtosis: {kurtosis(sem_scores):.4f}") # ~0 for normal (excess)