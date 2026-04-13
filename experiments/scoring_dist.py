#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 2026
@author: maximilianschulten
"""
#%% Setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.gen_utils import load_embedder
from lib.skills_utils import SkillsExtractor
from lib.scoring import skill_coverage, semantic_score, education_coverage, embed_doc
import numpy as np
import pandas as pd

ignore_cache = False

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

#%% Extract + embed
emb = load_embedder()

vars_ = ['total_scores', 'skill_scores', 'sem_scores', 'edu_scores', 'score_categories']

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
    total_scores    = []
    skill_scores    = []
    sem_scores      = []
    edu_scores      = []
    score_categories = []

    extractor = SkillsExtractor()

    # Pre-extract and embed all JDs
    print("Extracting + embedding job descriptions...")
    jd_skills_list   = []
    jd_edu_list      = []
    jd_skill_embs_list = []
    jd_doc_embs_list = []
    for idx, jdrow in jds.iterrows():
        jd_skills_raw, jd_edu = extractor.extract_all(jdrow['job_description'], use_gliner=True, use_spacy=True)
        jd_skills = list(jd_skills_raw.keys())
        jd_skills_list.append(jd_skills)
        jd_edu_list.append(jd_edu)
        jd_skill_embs_list.append(
            emb.encode(jd_skills, convert_to_numpy=True, normalize_embeddings=True) if jd_skills else None
        )
        jd_doc_embs, _ = embed_doc(jdrow['job_description'], embedding_model=emb, window_size=30, stride=10)
        jd_doc_embs_list.append(jd_doc_embs)
        if idx % 20 == 0:
            print(f"  JD {idx}/{len(jds)}")

    # Pre-extract and embed all resumes
    print("Extracting + embedding resumes...")
    res_skills_list     = []
    res_edu_list        = []
    res_skill_embs_list = []
    res_major_embs_list = []
    res_doc_embs_list   = []
    for jdx, resrow in resumes.iterrows():
        res_skills_raw, res_edu = extractor.extract_all(resrow['Text'], use_gliner=True, use_spacy=True)
        res_skills = list(res_skills_raw.keys())
        res_skills_list.append(res_skills)
        res_edu_list.append(res_edu)
        res_skill_embs_list.append(
            emb.encode(res_skills, convert_to_numpy=True, normalize_embeddings=True) if res_skills else None
        )
        res_majors = res_edu.get("majors", [])
        res_major_embs_list.append(
            emb.encode(res_majors, convert_to_numpy=True, normalize_embeddings=True) if res_majors else None
        )
        res_doc_embs, _ = embed_doc(resrow['Text'], embedding_model=emb, window_size=30, stride=10)
        res_doc_embs_list.append(res_doc_embs)

    # Pairwise scoring
    print("Computing pairwise scores...")
    alpha, beta_, gamma, tau = 0.45, 0.45, 0.1, 0.6

    for idx, (jd_skills, jd_edu, jd_skill_embs, jd_doc_embs) in enumerate(
        zip(jd_skills_list, jd_edu_list, jd_skill_embs_list, jd_doc_embs_list)
    ):
        for jdx, (res_skills, res_edu, res_skill_embs, res_major_embs, res_doc_embs) in enumerate(
            zip(res_skills_list, res_edu_list, res_skill_embs_list, res_major_embs_list, res_doc_embs_list)
        ):
            if jdx % 10 == 0:
                print("#", idx, '.', jdx, sep="")

            if jd_skills and jd_skill_embs is not None and res_skills and res_skill_embs is not None:
                sk, _, _ = skill_coverage(res_skills, jd_skills, res_skill_embs, jd_skill_embs, tau=tau)
            else:
                sk = 0.0

            sem = float(semantic_score(res_doc_embs, jd_doc_embs))

            edu = float(education_coverage(res_edu, jd_edu, emb, resume_major_embs=res_major_embs))

            total = alpha * sk + beta_ * sem + gamma * edu

            skill_scores.append(float(sk))
            sem_scores.append(float(sem))
            edu_scores.append(float(edu))
            total_scores.append(float(total))
            score_categories.append(resumes.iloc[jdx]['Category'])

    total_scores     = np.array(total_scores,     dtype=np.float32)
    skill_scores     = np.array(skill_scores,     dtype=np.float32)
    sem_scores       = np.array(sem_scores,       dtype=np.float32)
    edu_scores       = np.array(edu_scores,       dtype=np.float32)
    score_categories = np.array(score_categories)

#%% Save
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "total_scores.npy"),     total_scores)
np.save(os.path.join(out_dir, "skill_scores.npy"),     skill_scores)
np.save(os.path.join(out_dir, "sem_scores_full.npy"),  sem_scores)
np.save(os.path.join(out_dir, "edu_scores.npy"),       edu_scores)
np.save(os.path.join(out_dir, "score_categories.npy"), score_categories)
print(f"Saved {len(total_scores)} records to {out_dir}")

results = pd.DataFrame({
    'category':    score_categories,
    'total':       total_scores,
    'skill':       skill_scores,
    'semantic':    sem_scores,
    'education':   edu_scores,
})

#%% Distribution of composite scores
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, beta as beta_dist

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(total_scores, bins=40, edgecolor='black')
ax.set_title('Composite Score Distribution')
ax.set_xlabel('Score')
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()

#%% KDE of composite scores
fig, ax = plt.subplots(figsize=(7, 4))
data = np.array(total_scores)
nonzero = data[data > 0]
ax.hist(nonzero, bins=50, edgecolor='black', density=True, alpha=0.6)
kde = gaussian_kde(nonzero)
a, b, loc, scale = beta_dist.fit(nonzero)
m, s = norm.fit(nonzero)
xs = np.linspace(nonzero.min(), nonzero.max(), 300)
ax.plot(xs, kde(xs),                              linewidth=2, label='Gaussian KDE',                              linestyle='--', c='steelblue')
ax.plot(xs, beta_dist.pdf(xs, a, b, loc, scale),  linewidth=2, label=rf'$\alpha={a:.2f}$, $\beta={b:.2f}$',      linestyle='-.', c='navy')
ax.plot(xs, norm.pdf(xs, m, s),                   linewidth=2, label=rf'$\mu={m:.2f}$, $\sigma={s:.2f}$',        linestyle=':',  c='cornflowerblue')
ax.set_title('Distribution of Composite Scores')
ax.set_xlabel('Score')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()

#%% Side-by-side KDE of all three components
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

component_cfg = [
    (skill_scores,  'Skill Coverage',  'steelblue',      'navy',       'cornflowerblue'),
    (sem_scores,    'Semantic Score',  'maroon',         'red',        'coral'),
    (edu_scores,    'Education Score', 'seagreen',       'darkgreen',  'mediumseagreen'),
]

for ax, (scores, title, c_kde, c_beta, c_norm) in zip(axes, component_cfg):
    data = np.array(scores)
    nonzero = data[data > 0]
    ax.hist(nonzero, bins=40, edgecolor='black', density=True, alpha=0.6)
    kde = gaussian_kde(nonzero)
    a, b, loc, scale = beta_dist.fit(nonzero)
    m, s = norm.fit(nonzero)
    xs = np.linspace(nonzero.min(), nonzero.max(), 300)
    ax.plot(xs, kde(xs),                             linewidth=2, label='KDE',                            linestyle='--', c=c_kde)
    ax.plot(xs, beta_dist.pdf(xs, a, b, loc, scale), linewidth=2, label=rf'$\alpha={a:.2f}$, $\beta={b:.2f}$', linestyle='-.', c=c_beta)
    ax.plot(xs, norm.pdf(xs, m, s),                  linewidth=2, label=rf'$\mu={m:.2f}$, $\sigma={s:.2f}$',   linestyle=':',  c=c_norm)
    ax.set_title(f'Distribution of {title}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

#%% Per-category composite score distributions
fig, ax = plt.subplots(figsize=(12, 5))
for cat, group in results.groupby('category'):
    scores = group['total'].values
    if len(scores) >= 20:
        mu, std = norm.fit(scores)
        xs = np.linspace(scores.min(), scores.max(), 200)
        ax.plot(xs, norm.pdf(xs, mu, std), label=f"{cat} (μ={mu:.2f})")
ax.set_title('Per-category composite score distributions')
ax.set_xlabel('Composite score')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

#%% Per-category mean component breakdown
cat_means = results.groupby('category')[['skill', 'semantic', 'education', 'total']].mean()
cat_means = cat_means.sort_values('total', ascending=False)

x = np.arange(len(cat_means))
width = 0.2
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - 1.5 * width, cat_means['skill'],     width, label='Skill coverage',  color='steelblue')
ax.bar(x - 0.5 * width, cat_means['semantic'],  width, label='Semantic score',  color='coral')
ax.bar(x + 0.5 * width, cat_means['education'], width, label='Education score', color='seagreen')
ax.bar(x + 1.5 * width, cat_means['total'],     width, label='Composite score', color='slategrey')
ax.set_xticks(x)
ax.set_xticklabels(cat_means.index, rotation=45, ha='right', fontsize=8)
ax.set_title('Mean component scores by resume category')
ax.set_ylabel('Mean score')
ax.legend()
plt.tight_layout()
plt.show()

#%% Component correlation
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
pairs = [
    ('skill',    'semantic',   'Skill vs. Semantic'),
    ('skill',    'education',  'Skill vs. Education'),
    ('semantic', 'education',  'Semantic vs. Education'),
]
for ax, (cx, cy, title) in zip(axes, pairs):
    ax.scatter(results[cx], results[cy], alpha=0.15, s=4, c='steelblue')
    r = np.corrcoef(results[cx], results[cy])[0, 1]
    ax.set_title(f'{title}\n(r = {r:.3f})')
    ax.set_xlabel(cx.capitalize())
    ax.set_ylabel(cy.capitalize())
plt.tight_layout()
plt.show()

#%% Fit global params + stats
from scipy.stats import shapiro, skew, kurtosis

global_mu, global_std = norm.fit(total_scores)
print(f"Global composite fit: mu={global_mu:.4f}, std={global_std:.4f}")

print("\nPer-category fits (composite):")
for cat, group in results.groupby('category'):
    scores = group['total'].values
    if len(scores) >= 30:
        mu, std = norm.fit(scores)
        print(f"  {cat}: mu={mu:.4f}, std={std:.4f}")

print(f"\nShapiro-Wilk (composite): {shapiro(total_scores)}")
print(f"Skewness:  {skew(total_scores):.4f}")
print(f"Kurtosis:  {kurtosis(total_scores):.4f}")
