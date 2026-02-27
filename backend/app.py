import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BASE_DIR, MODEL_PATH
from flask import Flask, render_template, send_from_directory, request, jsonify
from utils.gen_utils import load_embedder, load_classifier, load_nlp
from utils.skills_utils import load_gliner
from utils.resume_utils import ResumeClassifier
from utils.skills_utils import SkillsExtractor, load_skills_matcher


app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")
FRONTEND = os.path.join(os.path.dirname(__file__), "../frontend")

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
skill_extractor = SkillsExtractor(nlp=nlp, matcher=matcher)
print("Skills Extractor loaded")
print("All Models loaded.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND, filename)

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    resume_text = data["resume_text"]
    cleaned_resume = classifier.clean_resume(resume_text)
    categories = classifier.classify_resume(cleaned_resume)
    skills = skill_extractor.extract_skills(cleaned_resume)
    return jsonify({
        "categories": categories,
        "skills": skills
    })


if __name__ == "__main__":
    app.run(debug=True)