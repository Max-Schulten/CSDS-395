import json
import re
from spacy.matcher import PhraseMatcher
from config import BASE_DIR
import os
from gliner import GLiNER
from utils.gen_utils import load_nlp

# Maps regex patterns (case-insensitive) to canonical degree level strings.
# Order matters — more specific / longer patterns must come before shorter ones
# (e.g. "ph.d" before bare "d", "m.b.a" before "m.a").
DEGREE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ── Doctorate ─────────────────────────────────────────────────────────────
    (re.compile(r"ph\.?\s*d\.?", re.I),                          "phd"),
    (re.compile(r"doctor\s+of\s+\w+", re.I),                     "phd"),
    (re.compile(r"docto(?:r|rate)", re.I),                        "phd"),
    (re.compile(r"d\.?\s*sc\.?", re.I),                          "phd"),

    # ── Master's ──────────────────────────────────────────────────────────────
    (re.compile(r"m\.?\s*b\.?\s*a\.?", re.I),                    "master's"),  # MBA before M.A.
    (re.compile(r"m\.?\s*[se]ng\.?", re.I),                      "master's"),  # M.Eng / M.S.Eng
    (re.compile(r"m\.?\s*s\.?\s*c\.?", re.I),                    "master's"),  # M.Sc before M.S.
    (re.compile(r"m\.?\s*[sa]\.?(?![a-z])", re.I),               "master's"),  # M.S. / M.A.
    (re.compile(r"master(?:'?s)?(?:\s+of\s+\w+)?", re.I),        "master's"),

    # ── Bachelor's ────────────────────────────────────────────────────────────
    (re.compile(r"b\.?\s*[se]ng\.?", re.I),                      "bachelor's"),  # B.Eng before B.E.
    (re.compile(r"b\.?\s*s\.?\s*c\.?(?![a-z])", re.I),            "bachelor's"),  # B.Sc before B.S.
    (re.compile(r"b\.?\s*[saef]\.?(?![a-z])", re.I),             "bachelor's"),  # B.S. / B.A. / B.E. / B.F.A.
    (re.compile(r"bachelor(?:'?s)?(?:\s+of\s+\w+)?", re.I),      "bachelor's"),

    # ── Associate's ───────────────────────────────────────────────────────────
    (re.compile(r"a\.?\s*a\.?\s*s\.?", re.I),                    "associate's"),  # A.A.S. before A.A./A.S.
    (re.compile(r"a\.?\s*[as]\.?", re.I),                        "associate's"),
    (re.compile(r"associate(?:'?s)?(?:\s+of\s+\w+)?", re.I),     "associate's"),

    # ── High school ───────────────────────────────────────────────────────────
    (re.compile(r"high\s+school\s+diploma", re.I),               "high school diploma"),
    (re.compile(r"\bged\b", re.I),                               "high school diploma"),
    (re.compile(r"h\.?\s*s\.?\s*d\.?", re.I),                   "high school diploma"),
]

def normalize_degree(raw: str) -> str:
    """Return the canonical degree level for a raw GLiNER-extracted string, or the
    lowercased original if no pattern matches."""
    for pattern, canonical in DEGREE_PATTERNS:
        if pattern.search(raw):
            return canonical
    return raw.lower().strip()


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
        self.gliner_labs = {"skills": ["job skill", "technical software", "technical ability", "certification"], "education": ["education degree", "education major"]}
        self.skills_map = skills_map
    
    def extract_skills(self, text: str) -> dict[str, list[int]]:
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
        gliner_matches = self.gliner.predict_entities(text, self.gliner_labs["skills"], threshold=0.5)
        for match in gliner_matches:
            if match['label'] != 'education':
                skill_text = match['text'].lower().strip()
                skill = self.skills_map.get(skill_text, skill_text)
                if skill not in seen:
                    seen.add(skill)
                    skills[skill] = [match['start'], match['end']]
        return skills
    
    _CANONICAL_DEGREES = {"phd", "master's", "bachelor's", "associate's", "high school diploma"}
    _DEGREE_ORDER = {"high school diploma": 0, "associate's": 1, "bachelor's": 2, "master's": 3, "phd": 4}
    # Higher-rank degrees require stronger confidence to guard against noise promoting degree level
    _DEGREE_CONFIDENCE_FLOOR = {"phd": 0.75, "master's": 0.75, "bachelor's": 0.5, "associate's": 0.5, "high school diploma": 0.5}

    def extract_education(self, text: str) -> dict:
        degree_candidates = []  # (rank, score, canonical)
        majors = []
        seen_majors = set()

        gliner_matches = self.gliner.predict_entities(text, self.gliner_labs["education"], threshold=0.5)
        for match in gliner_matches:
            if match['label'] != 'education degree':
                continue

            normalized = normalize_degree(match['text'])
            if normalized not in self._CANONICAL_DEGREES:
                continue  # school name or other noise — skip

            # Always extract majors from the span regardless of confidence.
            # Break after first matching pattern to avoid over-consuming adjacent words.
            raw = match['text'].strip(' ,.')
            remainder = raw
            for pattern, _ in DEGREE_PATTERNS:
                stripped = pattern.sub('', remainder).strip(' ,.')
                if stripped != remainder:
                    remainder = stripped
                    break
            major_lower = remainder.lower()
            if remainder and remainder != raw and major_lower not in seen_majors:
                majors.append(major_lower)
                seen_majors.add(major_lower)

            # Only trust this match for degree-level selection if it clears the confidence floor
            floor = self._DEGREE_CONFIDENCE_FLOOR[normalized]
            if match['score'] >= floor:
                degree_candidates.append((self._DEGREE_ORDER[normalized], match['score'], normalized))

        # Take highest rank; break ties by GLiNER confidence
        if degree_candidates:
            education_level = max(degree_candidates, key=lambda x: (x[0], x[1]))[2]
        else:
            education_level = None

        return {"degree": education_level, "majors": majors}
