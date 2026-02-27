import spacy
from config import BASE_DIR
import os
import joblib
from sentence_transformers import SentenceTransformer
from huggingface_hub import model_info
from huggingface_hub.errors import RepositoryNotFoundError

def load_nlp(model: str = "en_core_web_md", disable: list[str] = []):
    if not spacy.util.is_package(model):
        raise OSError(f"spaCy model '{model}' is not installed.")
    nlp = spacy.load(model, disable=disable)
    return nlp

def load_classifier(model_path: str = os.path.join(BASE_DIR, "models/resume_classifier.joblib")):
    if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at '{model_path}'.")
    model = joblib.load(model_path)
    return model

def load_embedder(model: str = "all-MiniLM-L6-v2"):
    try:
        model_info(f"sentence-transformers/{model}")
    except RepositoryNotFoundError:
        raise ValueError(f"'{model}' is not a valid SentenceTransformer model.")
    embedder = SentenceTransformer(model)
    return embedder