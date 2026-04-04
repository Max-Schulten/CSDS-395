from utils.skills_utils import SkillsExtractor, load_gliner
from utils.resume_utils import ResumeClassifier
from utils.gen_utils import load_embedder
from utils.resume import Resume
from utils.job import Job
from utils.scoring import score

# ── Sample data ───────────────────────────────────────────────────────────────

RESUME_TECH = """
John Doe | john@example.com
Skills: Python, machine learning, data analysis, SQL, PyTorch, scikit-learn
Education: B.S. Computer Science, State University, 2022
Experience: Data Science Intern at Acme Corp. Built ML pipelines using Python and scikit-learn.
Deployed models to production. Wrote SQL queries for feature engineering.
"""

RESUME_NONTEH = """
Sarah Chen | sarah@example.com
Skills: content strategy, copywriting, SEO, social media marketing, Google Analytics,
        email marketing, campaign management, brand development, market research
Education: B.A. Communications, Riverside University, 2020
Experience: Marketing Coordinator at BrightBrand Agency. Managed multi-channel campaigns
reaching 500k+ users. Wrote long-form content and managed editorial calendar.
Ran A/B tests on email subject lines, improving open rates by 18%.
Coordinated with external vendors and tracked KPIs using Google Analytics.
"""

JDS_TECH = {
    "ML Engineer": """
        We are looking for a machine learning engineer with strong Python skills.
        Experience with PyTorch or TensorFlow required. SQL and data pipeline experience a plus.
        B.S. in Computer Science or related field required.
    """,
    "Backend Engineer": """
        Backend software engineer role. Proficiency in Java or Go required.
        Experience with REST APIs, Kubernetes, and distributed systems.
        B.S. in Computer Science or equivalent experience.
    """,
    "Data Analyst": """
        Data analyst with strong SQL and Excel skills. Experience with Tableau or Power BI.
        Python a plus. B.S. in Statistics, Math, or related field.
    """,
}

JDS_NONTECH = {
    "Marketing Manager": """
        Seeking a marketing manager to lead brand strategy and digital campaigns.
        Strong copywriting and content strategy skills required. Experience with SEO,
        email marketing, and social media platforms. Proficiency in Google Analytics or
        similar tools. B.A. in Marketing, Communications, or related field.
    """,
    "Financial Analyst": """
        Financial analyst role focused on forecasting, budgeting, and variance analysis.
        Strong Excel and financial modeling skills required. Experience with SQL a plus.
        CPA or CFA designation preferred. B.S. in Finance, Accounting, or Economics required.
    """,
    "HR Generalist": """
        HR generalist responsible for recruiting, onboarding, employee relations, and
        benefits administration. Strong interpersonal and written communication skills.
        Experience with HRIS systems and performance management processes.
        B.A. in Human Resources, Psychology, or Business Administration required.
    """,
}

# ── Parameters ────────────────────────────────────────────────────────────────

PARAMS = dict(
    alpha=0.35,      # Skill coverage weight
    beta=0.5,        # Semantic similarity weight
    gamma=0.15,      # Education coverage weight
    tau=0.8,         # Skill match threshold (higher = stricter)
    window_size=30,  # Token window for semantic embedding
    stride=10,       # Window stride (smaller = more overlap)
)

# ── Runner ────────────────────────────────────────────────────────────────────

def run(resume_text=RESUME_TECH, jd_dict=JDS_TECH, embedding_model=None, extractor=None, gliner=None, label="", **params):
    p = {**PARAMS, **params}
    embedding_model = embedding_model if embedding_model is not None else load_embedder()
    extractor = extractor if extractor is not None else SkillsExtractor()
    gliner = gliner if gliner is not None else load_gliner()
    resume = Resume(resume_text, classifier=ResumeClassifier(), skill_extractor=extractor, gliner=gliner)
    jobs   = [Job(jd, skill_extractor=extractor) for jd in jd_dict.values()]
    names  = list(jd_dict.keys())

    results = score(resume, jobs, embedding_model=embedding_model, **p)

    header = f"  {label}  " if label else ""
    print(f"\n{'='*60}")
    print(f"{header}alpha={p['alpha']}  beta={p['beta']}  gamma={p['gamma']}")
    print(f"  tau={p['tau']}  window={p['window_size']}  stride={p['stride']}")
    print(f"{'='*60}")
    print(f"  {'Job':<24} {'Skill':>6} {'Sem':>6} {'Edu':>6} {'Score':>7}")
    print(f"  {'-'*24} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for i, name in enumerate(names):
        sk  = results["skill_coverages"][i]
        sem = results["semantic_scores"][i]
        edu = results["education_coverages"][i]
        s   = results["scores"][i]
        print(f"  {name:<24} {sk:>6.3f} {sem:>6.3f} {edu:>6.3f} {s:>7.3f}")
    print()
    return results


if __name__ == "__main__":
    e=load_embedder()
    ext = SkillsExtractor()
    g = load_gliner()

    # ── Education extraction debug ────────────────────────────────────────────
    from utils.resume_utils import ResumeClassifier
    from utils.resume import Resume
    clf = ResumeClassifier()
    print("\n[DEBUG] Tech resume:    ", Resume(RESUME_TECH,   classifier=clf, skill_extractor=ext, gliner=g).education)
    print("[DEBUG] Non-tech resume:", Resume(RESUME_NONTEH, classifier=clf, skill_extractor=ext, gliner=g).education)
    print("\n[DEBUG] Tech resume:    ", Resume(RESUME_TECH,   classifier=clf, skill_extractor=ext, gliner=g).skills)
    print("[DEBUG] Non-tech resume:", Resume(RESUME_NONTEH, classifier=clf, skill_extractor=ext, gliner=g).skills)

    # ── Default params, all resume/JD combos ─────────────────────────────────
    run(RESUME_TECH,   JDS_TECH,    label="[Tech resume    × Tech JDs]   ", embedding_model=e, extractor=ext, gliner=g)
    run(RESUME_TECH,   JDS_NONTECH, label="[Tech resume    × Non-tech JDs]", embedding_model=e, extractor=ext, gliner=g)
    run(RESUME_NONTEH, JDS_TECH,    label="[Non-tech resume × Tech JDs]   ", embedding_model=e, extractor=ext, gliner=g)
    run(RESUME_NONTEH, JDS_NONTECH, label="[Non-tech resume × Non-tech JDs]", embedding_model=e, extractor=ext, gliner=g)

    # ── Tau sweep (tech resume × tech JDs) ───────────────────────────────────
    for tau in [0.6, 0.7, 0.8, 0.9]:
        run(RESUME_TECH, JDS_TECH, label=f"[tau sweep]", tau=tau, embedding_model=e, extractor=ext, gliner=g)

    # ── Weight sweep (tech resume × tech JDs) ────────────────────────────────
    for alpha, beta, gamma in [(0.5, 0.3, 0.2), (0.2, 0.6, 0.2), (0.33, 0.34, 0.33)]:
        run(RESUME_TECH, JDS_TECH, label="[weight sweep]", alpha=alpha, beta=beta, gamma=gamma, embedding_model=e, extractor=ext, gliner=g)

    # ── Window/stride sweep (tech resume × tech JDs) ─────────────────────────
    for window_size, stride in [(20, 5), (30, 10), (50, 20)]:
        run(RESUME_TECH, JDS_TECH, label="[window sweep]", window_size=window_size, stride=stride, embedding_model=e, extractor=ext, gliner=g)
