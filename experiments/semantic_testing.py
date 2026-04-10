#%% Setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.gen_utils import load_embedder
from lib.scoring import semantic_score, embed_doc
import numpy as np

#%% Embs
emb = load_embedder()

with open('strong_resume.txt', 'r') as file: strong_resume = file.read()
with open('weak_resume.txt', 'r') as file: weak_resume = file.read()
with open('job_desc.txt', 'r') as file: job_desc = file.read()


emb_job , _ = embed_doc(job_desc, emb, window_size=30)
emb_weak, _ = embed_doc(weak_resume, emb, window_size=30)
emb_strong, _ = embed_doc(strong_resume, emb, window_size=30)
#%% Messing around

sem_score_strong = semantic_score(emb_strong, emb_job) # We want a value close to 1, i.e. [0.85/0.9,1]
print("STRONG:",sem_score_strong)

sem_score_weak = semantic_score(emb_weak, emb_job) # We want a value close to 1, i.e. [0.85/0.9,1]
print("WEAK:",sem_score_weak)
