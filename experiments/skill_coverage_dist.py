#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 2026
@author: maximilianschulten
"""
#%% Setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.gen_utils import load_embedder
from lib.skills_utils import SkillsExtractor
from lib.scoring import skill_coverage
import numpy as np
import pandas as pd

ignore_cache = True

out_dir = os.path.join(os.path.dirname(__file__), "dist_data")

#%% Data
all_resumes = pd.read_csv(os.path.join(os.path.dirname(__file__), "..") + "/data/resume-class.csv")
jds = pd.read_csv(os.path.join(os.path.dirname(__file__), "..") + "/data/job-descriptions.csv").sample(n=200,
                                                                                                        ignore_index=True,
                                                                                                        random_state=4)
n_per_cat = 3
resumes = pd.concat([
    g.sample(n=min(len(g), n_per_cat), random_state=42)
    for _, g in all_resumes.groupby("Category")
]).reset_index(drop=True)
print(f"Stratified resume sample: {len(resumes)} resumes across {resumes['Category'].nunique()} categories")

#%% Extract skills + compute coverage
emb = load_embedder()

vars_ = ['cov_scores', 'cov_categories', 'n_job_skills_arr', 'n_covered_arr']

n_cache_hits = 0

for var in vars_:
    dir_ = os.path.join(out_dir, f"{var}.npy")
    print(dir_)
    if not os.path.exists(dir_):
        print("Miss")
        continue
    print("hit")
    data = np.load(dir_, allow_pickle=True)
    exec(f"{var}=data")
    n_cache_hits += 1

cached = n_cache_hits == len(vars_)

if not cached or ignore_cache:
    cov_scores       = []
    cov_categories   = []
    n_job_skills_arr = []
    n_covered_arr    = []
    extractor = SkillsExtractor()

    # Extract skills once per document to avoid redundant model inference
    print("Extracting job description skills...")
    jd_skills_list = []
    jd_embs_list   = []
    for idx, jdrow in jds.iterrows():
        jd_skills = list(extractor.extract_skills(jdrow['job_description'], use_spacy=True).keys())
        jd_skills_list.append(jd_skills)
        if jd_skills:
            jd_embs_list.append(emb.encode(jd_skills, convert_to_numpy=True, normalize_embeddings=True))
        else:
            jd_embs_list.append(None)
        if idx % 20 == 0:
            print(f"  JD {idx}/{len(jds)}")

    print("Extracting resume skills...")
    res_skills_list = []
    res_embs_list   = []
    for jdx, resrow in resumes.iterrows():
        res_skills = list(extractor.extract_skills(resrow['Text'], use_spacy=True).keys())
        res_skills_list.append(res_skills)
        if res_skills:
            res_embs_list.append(emb.encode(res_skills, convert_to_numpy=True, normalize_embeddings=True))
        else:
            res_embs_list.append(None)

    print("Computing pairwise skill coverage...")
    for idx, (jd_skills, jd_embs) in enumerate(zip(jd_skills_list, jd_embs_list)):
        for jdx, (res_skills, res_embs) in enumerate(zip(res_skills_list, res_embs_list)):
            if jdx % 10 == 0:
                print("#", idx, '.', jdx, sep="")

            if not jd_skills or jd_embs is None or not res_skills or res_embs is None:
                cov, covered = 0.0, []
            else:
                cov, covered, _ = skill_coverage(res_skills, jd_skills, res_embs, jd_embs)

            cov_scores.append(cov)
            cov_categories.append(resumes.iloc[jdx]['Category'])
            n_job_skills_arr.append(len(jd_skills))
            n_covered_arr.append(len(covered))

    cov_scores       = np.array(cov_scores,       dtype=np.float32)
    cov_categories   = np.array(cov_categories)
    n_job_skills_arr = np.array(n_job_skills_arr, dtype=np.int32)
    n_covered_arr    = np.array(n_covered_arr,    dtype=np.int32)

#%% Save
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "cov_scores.npy"),       cov_scores)
np.save(os.path.join(out_dir, "cov_categories.npy"),   cov_categories)
np.save(os.path.join(out_dir, "n_job_skills_arr.npy"), n_job_skills_arr)
np.save(os.path.join(out_dir, "n_covered_arr.npy"),    n_covered_arr)
print(f"Saved {len(cov_scores)} records to {out_dir}")

results = pd.DataFrame({
    'category':     cov_categories,
    'cov_score':    cov_scores,
    'n_job_skills': n_job_skills_arr,
    'n_covered':    n_covered_arr,
})

#%% Distribution approximation
import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(15, 4))
axes.hist(cov_scores, bins=30, edgecolor='black')
axes.set_title('Skill Coverage Scores')
axes.set_xlabel('Score')
axes.set_ylabel('Count')
plt.tight_layout()
plt.show()

#%% KDE
from scipy.stats import gaussian_kde, norm, beta as beta_dist
fig, axes = plt.subplots(figsize=(7, 4))
data = np.array(cov_scores)
nonzero = data[data != 0]
axes.hist(nonzero, bins=50, edgecolor='black', density=True, alpha=0.6)
kde = gaussian_kde(nonzero)
a, b, loc, scale = beta_dist.fit(nonzero)
m, s = norm.fit(nonzero)
xs = np.linspace(nonzero.min(), nonzero.max(), 300)
axes.plot(xs, kde(xs), linewidth=2, label="Gaussian KDE Estimate", linestyle='--', c='steelblue')
axes.plot(xs, beta_dist.pdf(xs, a, b, loc, scale), linewidth=2, label=rf'$\alpha={a:.2f}$, $\beta={b:.2f}$', linestyle='-.', c='navy')
axes.plot(xs, norm.pdf(xs, m, s), linewidth=2, label=rf'$\mu={m:.2f}$, $\sigma={s:.2f}$', linestyle=':', c='cornflowerblue')
axes.set_title('Distribution of Skill Coverage Scores')
axes.set_xlabel('Score')
axes.set_ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

#%% Coverage vs. number of job skills
n_buckets = 15
bucket_means   = []
bucket_stds    = []
bucket_centers = []
jsk = n_job_skills_arr
lo_, hi_ = jsk.min(), jsk.max()
edges = np.linspace(lo_, hi_, n_buckets + 1)
for i in range(n_buckets):
    mask = (jsk >= edges[i]) & (jsk < edges[i + 1])
    if mask.sum() > 0:
        bucket_means.append(cov_scores[mask].mean())
        bucket_stds.append(cov_scores[mask].std())
        bucket_centers.append((edges[i] + edges[i + 1]) / 2)

bucket_means   = np.array(bucket_means)
bucket_stds    = np.array(bucket_stds)
bucket_centers = np.array(bucket_centers)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(bucket_centers, bucket_means, linewidth=2, label='Mean coverage')
ax.fill_between(bucket_centers,
                bucket_means - bucket_stds,
                bucket_means + bucket_stds,
                alpha=0.2, label='±1 std')
ax.set_title('Mean skill coverage by number of job skills required')
ax.set_xlabel('Number of job skills')
ax.set_ylabel('Mean coverage score')
ax.legend()
plt.tight_layout()
plt.show()

#%% Per-category distribution
fig, ax = plt.subplots(figsize=(12, 5))
for cat, group in results.groupby('category'):
    scores = group['cov_score'].values
    if len(scores) >= 20:
        mu, std = norm.fit(scores)
        xs = np.linspace(scores.min(), scores.max(), 200)
        ax.plot(xs, norm.pdf(xs, mu, std), label=f"{cat} (μ={mu:.2f})")
ax.set_title('Per-category skill coverage distributions')
ax.set_xlabel('Skill coverage score')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

#%% Fit global params
global_mu, global_std = norm.fit(cov_scores)
print(f"Global fit: mu={global_mu:.4f}, std={global_std:.4f}")

category_params = {}
for cat, group in results.groupby('category'):
    scores = group['cov_score'].values
    if len(scores) >= 30:
        mu, std = norm.fit(scores)
        category_params[cat] = (mu, std)
        print(f"{cat}: mu={mu:.4f}, std={std:.4f}")

#%%
from scipy.stats import shapiro, skew, kurtosis
print(shapiro(cov_scores))
print(f"Skewness: {skew(cov_scores):.4f}")
print(f"Kurtosis: {kurtosis(cov_scores):.4f}")
