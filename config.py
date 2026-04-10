import os
from dotenv import load_dotenv

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/resume_classifier.joblib")

# API ENDPOINT
HEADERS = {"User-Agent": "SkillSync/1.0"}
REQUEST_DELAY = 0.2

load_dotenv()
# API KEYS
FINDWORK_API_KEY   = os.getenv("FINDWORK_API_KEY", "")
USAJOBS_API_KEY    = os.getenv("USAJOBS_API_KEY", "")
USAJOBS_USER_AGENT = os.getenv("USAJOBS_USER_AGENT", "")
ADZUNA_APP_ID      = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY     = os.getenv("ADZUNA_APP_KEY", "")
THEMUSE_KEY        = os.getenv("THEMUSE_KEY", "")