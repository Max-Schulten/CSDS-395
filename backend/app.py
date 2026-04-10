import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BASE_DIR, MODEL_PATH
from flask import Flask, render_template, send_from_directory, request, jsonify, Response, stream_with_context
from lib.gen_utils import load_embedder, load_classifier, load_nlp, clean_text
from lib.skills_utils import load_gliner
from lib.resume_utils import ResumeClassifier
from lib.skills_utils import SkillsExtractor, load_skills_matcher, load_skills_map
from lib.resume import Resume
from lib.job import Job
from lib.jobfinder import JobFinder
from lib.scoring import score


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "frontend"), static_folder=os.path.join(BASE_DIR, "frontend"))
FRONTEND = os.path.join(BASE_DIR, "frontend")

print("Loading Skills Map...")
skills_map = load_skills_map(os.path.join(BASE_DIR, "data/skill_map.json"))
print("Skills map loaded.")

print("Loading Models...")
nlp = load_nlp()
print("NLP Loaded")
embedding_model = load_embedder()
print("Embedder loaded")
gliner = load_gliner()
print("GliNER loaded")
classifier = ResumeClassifier(model=load_classifier(), nlp_model=nlp)
print("SVC Loaded")
matcher = load_skills_matcher(nlp_model=nlp)
print("Skills Matcher Loaded")
skill_extractor = SkillsExtractor(nlp=nlp, matcher=matcher, gliner=gliner, skills_map=skills_map)
print("Skills Extractor loaded")
print("All Models loaded.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND, filename)

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

@app.route("/find-jobs", methods=["POST"])
def search_jobs():
    data = request.get_json()
    resume_text = data["resume_text"]
    n_jobs = int(data.get("n_jobs", 10))
    resume = Resume(
        resume_text=resume_text,
        classifier=classifier,
        skill_extractor=skill_extractor
    )
    jf = JobFinder(resume=resume, skill_extractor=skill_extractor, n_jobs=n_jobs)

    def generate():
        yield _sse({"type": "status", "message": "Fetching listings from external sources…"})

        sample = jf._fetch_and_rank()
        n = len(sample)
        yield _sse({"type": "status", "message": f"Found {n} ranked opportunit{'y' if n == 1 else 'ies'}. Extracting skills…"})

        jobs: list[Job] = []
        for idx, r in enumerate(sample, 1):
            title = r.get("job_title") or "opportunity"
            yield _sse({"type": "status", "message": f"Analyzing {idx}/{n}: {title}"})
            job = jf._build_job(r)
            if job:
                jobs.append(job)

        yield _sse({"type": "status", "message": "Scoring opportunities against your resume…"})
        result = score(resume=resume, jobs=jobs, embedding_model=embedding_model)
        yield _sse({"type": "result", **result})

    return Response(stream_with_context(generate()), content_type="text/event-stream")

@app.route("/score-job", methods=["POST"])
def score_job():
    data = request.get_json()
    resume_text = data["resume_text"]
    resume = Resume(
        resume_text=resume_text,
        classifier=classifier,
        skill_extractor=skill_extractor
    )
    job_text = data["job_text"]
    job = Job(
        job_desc=job_text,
        skill_extractor=skill_extractor
    )
    
    return score(resume=resume, jobs=job, embedding_model=embedding_model)


if __name__ == "__main__":
    app.run(debug=True)