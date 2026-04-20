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


@app.errorhandler(400)
def bad_request(e):
    app.logger.warning("400 Bad Request: %s", e)
    return jsonify({"error": "Your request could not be understood. Please check your input and try again."}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "The requested resource was not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "This action is not allowed."}), 405

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.exception("Unhandled exception: %s", e)
    return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

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
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON."}), 400
    if not data.get("resume_text") or not str(data["resume_text"]).strip():
        return jsonify({"error": "Resume text is required."}), 400

    try:
        resume_text = str(data["resume_text"])
        n_jobs = int(data.get("n_jobs", 10))
        resume = Resume(resume_text=resume_text, classifier=classifier, skill_extractor=skill_extractor)
        jf = JobFinder(resume=resume, skill_extractor=skill_extractor, n_jobs=n_jobs)
    except (ValueError, TypeError) as e:
        app.logger.error("Setup error in /find-jobs: %s", e)
        return jsonify({"error": "We could not process your resume. Please check that your file uploaded correctly and try again."}), 422

    def generate():
        try:
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
        except Exception as e:
            app.logger.exception("Streaming error in /find-jobs: %s", e)
            yield _sse({"type": "error", "message": "Something went wrong while searching for jobs. Please try again."})

    return Response(stream_with_context(generate()), content_type="text/event-stream")

@app.route("/score-job", methods=["POST"])
def score_job():
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON."}), 400
    if not data.get("resume_text") or not str(data["resume_text"]).strip():
        return jsonify({"error": "Resume text is required."}), 400
    if not data.get("job_text") or not str(data["job_text"]).strip():
        return jsonify({"error": "Job description is required."}), 400

    try:
        resume = Resume(resume_text=str(data["resume_text"]), classifier=classifier, skill_extractor=skill_extractor)
        job = Job(job_desc=str(data["job_text"]), skill_extractor=skill_extractor)
        return score(resume=resume, jobs=job, embedding_model=embedding_model)
    except (ValueError, TypeError) as e:
        app.logger.error("Input error in /score-job: %s", e)
        return jsonify({"error": "We could not process your resume or job description. Please check that both were pasted correctly and try again."}), 422
    except Exception as e:
        app.logger.exception("Unexpected error in /score-job: %s", e)
        return jsonify({"error": "An unexpected error occurred while scoring. Please try again later."}), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
