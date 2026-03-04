import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BASE_DIR, MODEL_PATH
from flask import Flask, render_template, send_from_directory, request, jsonify
from utils.gen_utils import load_embedder, load_classifier, load_nlp, clean_text
from utils.skills_utils import load_gliner
from utils.resume_utils import ResumeClassifier
from utils.skills_utils import SkillsExtractor, load_skills_matcher, load_skills_map


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
classifier = ResumeClassifier(model=load_classifier(), nlp_model=nlp, embedding_model=embedding_model)
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

@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    data = request.get_json()
    resume_text = data["resume_text"]
    cleaned_resume = classifier.clean_resume(resume_text)
    categories = classifier.classify_resume(cleaned_resume)
    skills = skill_extractor.extract_skills(cleaned_resume)
    return jsonify({
        "cleaned_resume": cleaned_resume,
        "education": "",
        "categories": categories,
        "skills": skills
    })

@app.route("/analyze-job", methods=["POST"])
def analyze_job():
    data = request.get_json()
    job_desc = data["job_desc"]
    clean_job_desc = clean_text(job_desc)
    skills = skill_extractor.extract_skills(clean_job_desc)
    return jsonify({
        "cleaned_job_desc": clean_job_desc,
        "skills": skills,
        "education": ""
    })


if __name__ == "__main__":
    app.run(debug=True)