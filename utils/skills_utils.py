import json
from spacy.matcher import PhraseMatcher
from config import BASE_DIR
import os
from gliner import GLiNER
from utils.gen_utils import load_nlp


def load_skills_map(skill_map_path: str):
    try:
        with open(skill_map_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Skills map json was not found at {skill_map_path}")

def load_skills_matcher(nlp_model = None,
                    skill_map_path: str = os.path.join(BASE_DIR, 'data/skill_map.json')):
    nlp_model = nlp_model if nlp_model is not None else load_nlp()
    skills_map = load_skills_map(skill_map_path)
    matcher = PhraseMatcher(nlp_model.vocab, attr="LOWER")  
    patterns = [nlp_model.make_doc(k) for k in set(skills_map.keys())]
    matcher.add("SKILL", patterns)
    return matcher
    
def load_gliner(model_name: str = "urchade/gliner_small-v2.1"):
    model = GLiNER.from_pretrained(model_name)
    return model

class SkillsExtractor:
    def __init__(self, nlp = None, matcher = None) -> None:
        nlp = nlp if nlp is not None else load_nlp()
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer", first=True)
        self.nlp = nlp
        self.matcher = matcher if matcher is not None else load_skills_matcher(nlp_model=nlp)
    
    def extract_skills(self, text: str):
        doc = self.nlp(text)
        seen = set()
        results = {}
        matches = self.matcher(doc) # type: ignore
        for _, start, end in matches:
            skill = doc[start:end].text.lower()
            if skill not in seen:
                seen.add(skill)
                results[skill] = doc[start].sent.text.strip()
        return results