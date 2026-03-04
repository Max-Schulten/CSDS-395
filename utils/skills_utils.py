import json
from spacy.matcher import PhraseMatcher
from config import BASE_DIR
import os
from gliner import GLiNER
from utils.gen_utils import load_nlp


def load_skills_map(skill_map_path: str = os.path.join(BASE_DIR, 'data/skill_map.json')):
    try:
        with open(skill_map_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Skills map json was not found at {skill_map_path}")

def load_skills_matcher(nlp_model = None, skills_map = None):
    skills_map = skills_map if skills_map is not None else load_skills_map()
    nlp_model = nlp_model if nlp_model is not None else load_nlp()
    matcher = PhraseMatcher(nlp_model.vocab, attr="LOWER")  
    patterns = [nlp_model.make_doc(k) for k in set(skills_map.keys())]
    matcher.add("SKILL", patterns)
    return matcher
    
def load_gliner(model_name: str = "gliner-community/gliner_medium-v2.5"):
    model = GLiNER.from_pretrained(model_name)
    return model

class SkillsExtractor:
    def __init__(self, nlp = None, matcher = None, gliner = None, skills_map = None) -> None:
        nlp = nlp if nlp is not None else load_nlp()
        skills_map = skills_map if skills_map is not None else load_skills_map()
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer", first=True)
        self.nlp = nlp
        self.matcher = matcher if matcher is not None else load_skills_matcher(nlp_model=nlp)
        self.gliner = gliner if gliner is not None else load_gliner()
        self.gliner_labs = ["skill", "software", "technical concept", "education"]
        self.skills_map = skills_map
    
    def extract_skills(self, text: str):
        doc = self.nlp(text)
        seen = set()
        skills = {} # Keys are skills, and values are indices of detection
        matches = self.matcher(doc) # type: ignore
        for _, start, end in matches:
            skill_text = doc[start:end].text.lower()
            skill = self.skills_map[skill_text]
            if skill not in seen:
                seen.add(skill)
                skills[skill] = [start, end]
        gliner_matches = self.gliner.predict_entities(text, self.gliner_labs, threshold = 0.5)
        for match in gliner_matches:
            if match['label'] in ["skill", "software", "technical concept"]:
                skill_text = match['text'].lower().strip()
                skill = self.skills_map.get(skill_text, skill_text)
                if skill not in seen:
                    seen.add(skill)
                    skills[skill] = [match['start'], match['end']]
        return skills
    
    def extract_education(self, text: str):
        return {}