from utils.resume import Resume
from utils.job import Job
from utils.gen_utils import load_embedder
import numpy as np

def score(resume: Resume, jobs: list[Job] | Job, embedding_model = None, alpha=0.45, beta=0.45, gamma=0.1, tau=0.8, window_size=30, stride=10) -> dict[str, str | list[str] | list[float]]:
    embedding_model = embedding_model if embedding_model is not None else load_embedder()
    out = {
        "resume": resume.to_dict(),
        "jobs": [],
        "skill_coverages": [],
        "semantic_scores": [],
        "education_coverages": [],
        "scores": []
    }
    
    # Coefficients 
    
    assert round(alpha + beta + gamma) == 1 # Ensure combination is convex
    
    resume_skills = list(resume.skills.keys())
    emb_res_skills = embedding_model.encode(resume_skills, convert_to_numpy=True, normalize_embeddings=True)

    res_majors = resume.education.get("majors", [])
    emb_res_majors = embedding_model.encode(res_majors, convert_to_numpy=True, normalize_embeddings=True) if res_majors else None

    res_embs, res_snaps = embed_doc(resume.resume_raw, embedding_model, window_size=window_size, stride=stride)
    
    if not isinstance(jobs, list):
        jobs = [jobs]
    
    for job in jobs:
        job_skills = list(job.skills.keys()) # type: ignore
        out['jobs'].append(job.to_dict())
        emb_job_skills = embedding_model.encode(job_skills, convert_to_numpy=True, normalize_embeddings=True)
        
        # Skill coverage score; in [0,1]
        skill_cov = skill_coverage(resume_skills, job_skills, emb_res_skills, emb_job_skills, tau)
        out["skill_coverages"].append(skill_cov)
        
        # Semantic score; in [0,1]
        job_embs, job_snaps = embed_doc(job.job_desc, embedding_model=embedding_model, window_size=window_size, stride=stride)
        sem_score = semantic_score(res_embs, job_embs)
        out["semantic_scores"].append(sem_score)
        
        # Education Coverage; in [0,1]
        ed_cov = education_coverage(resume.education, job.education, embedding_model, emb_res_majors)
        out["education_coverages"].append(ed_cov)
        
        total_score = alpha * skill_cov + beta * sem_score + gamma * ed_cov
        out["scores"].append(total_score)
        
    return out

def semantic_score(resume_embs, job_embs):
    sim_mat = job_embs @ resume_embs.T # type: ignore
    return sim_mat.max(axis=1).mean()
    
def skill_coverage(resume_skills, job_skills, resume_skills_emb, job_skills_emb, tau=0.5):
    sim_matrix = resume_skills_emb @ job_skills_emb.T
    match_mask = sim_matrix >= tau
    
    matched_pairs = [
        (resume_skills[i], job_skills[j], float(sim_matrix[i, j]))
        for i, j in zip(*np.where(match_mask))
    ]

    job_skill_scores = {}
    for rs, js, score_ in matched_pairs:
        # Keep the best score if a job skill is matched by multiple resume skills
        if js not in job_skill_scores or score_ > job_skill_scores[js]:
            job_skill_scores[js] = score_

    def weight(score_, tau):
        return (score_ - tau) / (1.0 - tau)  # 0 at tau, 1 at perfect match

    coverage_score = (
        sum(weight(s, tau) for s in job_skill_scores.values()) / len(job_skills)
        if job_skills else 0.0
    )
    return coverage_score

def education_coverage(resume_edu: dict, job_edu: dict, embedding_model, resume_major_embs=None, edu_tau=0.85):
    # ── Degree level ──────────────────────────────────────────────────────────
    DEGREE_ORDER = {"high school diploma": 0, "associate's": 1, "bachelor's": 2, "master's": 3, "phd": 4}

    res_degree = resume_edu.get("degree")
    job_degree = job_edu.get("degree")

    if job_degree is None:
        degree_score = 1.0                          # no requirement stated
    elif res_degree is None:
        degree_score = 0.5                          # can't determine — don't zero out
    else:
        res_rank = DEGREE_ORDER.get(res_degree, -1)
        job_rank = DEGREE_ORDER.get(job_degree, -1)
        if res_rank >= job_rank:
            degree_score = 1.0
        else:
            gap = job_rank - res_rank
            degree_score = 0.5 if gap == 1 else 0.0  # one level under: partial; two+: no credit

    # ── Major similarity ──────────────────────────────────────────────────────
    job_majors = job_edu.get("majors", [])

    if not job_majors:
        major_score = 1.0                           # no major required
    elif resume_major_embs is None:
        major_score = 0.5                           # resume major unknown
    else:
        emb_job_majors = embedding_model.encode(job_majors, convert_to_numpy=True, normalize_embeddings=True)
        sim_matrix = resume_major_embs @ emb_job_majors.T  # (n_res, n_job)
        best_per_job = sim_matrix.max(axis=0)              # best resume match per job major
        matched = (best_per_job >= edu_tau).sum()
        major_score = matched / len(job_majors)

    return 0.7 * degree_score + 0.3 * major_score


# Convolves a window of ctx_window with stride over text and embeds each window
def embed_doc(text, embedding_model, window_size=30, stride=10):
    tokens = text.split()

    snapshots = [
        " ".join(tokens[i:i+window_size])
        for i in range(0, len(tokens) - window_size + 1, stride)
    ]

    if not snapshots:
        snapshots = [text]

    return embedding_model.encode(snapshots, convert_to_numpy=True, normalize_embeddings=True), snapshots