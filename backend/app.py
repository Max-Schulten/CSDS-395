import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BASE_DIR, MODEL_PATH
from flask import Flask, render_template, send_from_directory, request, jsonify
from utils.resume_utils import ResumeClassifier

app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")
FRONTEND = os.path.join(os.path.dirname(__file__), "../frontend")

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
    
    RC = ResumeClassifier(
        model_dir=MODEL_PATH
    )

    cleaned_resume = RC.clean_resume(resume_text)
    results = RC.classify_resume(cleaned_resume)
    
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)