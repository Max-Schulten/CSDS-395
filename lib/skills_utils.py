import json
import re
from spacy.matcher import PhraseMatcher
from config import BASE_DIR
import os
from gliner import GLiNER
from lib.gen_utils import load_nlp

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
    _SKILL_LABELS = {"job skill", "technical software", "technical ability"}
    _EDUCATION_LABELS = {"education degree", "education major"}
    _ALL_LABELS = list(_SKILL_LABELS | _EDUCATION_LABELS)

    _CANONICAL_DEGREES = {"phd", "master's", "bachelor's", "associate's", "high school diploma"}
    _DEGREE_ORDER = {"high school diploma": 0, "associate's": 1, "bachelor's": 2, "master's": 3, "phd": 4}
    _DEGREE_CONFIDENCE_FLOOR = {"phd": 0.75, "master's": 0.75, "bachelor's": 0.5, "associate's": 0.5, "high school diploma": 0.5}

    def __init__(self, nlp = None, matcher = None, gliner = None, skills_map = None) -> None:
        nlp = nlp if nlp is not None else load_nlp()
        skills_map = skills_map if skills_map is not None else load_skills_map()
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer", first=True)
        self.nlp = nlp
        self.matcher = matcher if matcher is not None else load_skills_matcher(nlp_model=nlp)
        self.gliner = gliner if gliner is not None else load_gliner()
        self.skills_map = skills_map

    def _predict_entities_chunked(self, text: str, threshold: float = 0.7, chunk_size: int = 1000) -> list[dict]:
        """Run a single GLiNER pass over all labels in sentence-based chunks to avoid token-limit truncation."""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        all_entities = []
        chunk_sents: list = []
        chunk_char_start = 0

        def process_chunk(chunk_text: str, abs_start: int):
            for ent in self.gliner.predict_entities(chunk_text, self._ALL_LABELS, threshold=threshold):
                adjusted = dict(ent)
                adjusted['start'] += abs_start
                adjusted['end'] += abs_start
                all_entities.append(adjusted)

        def flush(sents, start_char):
            if not sents:
                return
            chunk_text = text[start_char:sents[-1].end_char]
            if len(chunk_text) <= chunk_size:
                process_chunk(chunk_text, start_char)
            else:
                # Sentence exceeds chunk_size — sub-split at word boundaries
                offset = 0
                while offset < len(chunk_text):
                    end = min(offset + chunk_size, len(chunk_text))
                    if end < len(chunk_text):
                        while end > offset and chunk_text[end] not in (' ', '\n', '\t'):
                            end -= 1
                        if end == offset:  # no whitespace found — hard cut
                            end = min(offset + chunk_size, len(chunk_text))
                    process_chunk(chunk_text[offset:end], start_char + offset)
                    offset = end

        for sent in sentences:
            if chunk_sents and (sent.end_char - chunk_char_start) > chunk_size:
                flush(chunk_sents, chunk_char_start)
                chunk_char_start = sent.start_char
                chunk_sents = [sent]
            elif chunk_sents:
                chunk_sents.append(sent)
            else:
                chunk_char_start = sent.start_char
                chunk_sents = [sent]

        flush(chunk_sents, chunk_char_start)
        return all_entities

    def extract_all(self, text: str, use_gliner: bool = True, use_spacy: bool = True) -> tuple[dict[str, list[int]], dict]:
        """Extract skills and education in a single GLiNER pass. Prefer this over calling
        extract_skills and extract_education separately to avoid redundant model inference."""
        doc = self.nlp(text)
        seen: set[str] = set()
        skills: dict[str, list[int]] = {}

        if use_spacy:
            for _, start, end in self.matcher(doc):  # type: ignore
                skill_text = doc[start:end].text.lower()
                skill = self.skills_map.get(skill_text, skill_text)
                if skill not in seen:
                    seen.add(skill)
                    skills[skill] = [start, end]

        degree_candidates: list[tuple[int, float, str]] = []
        majors: list[str] = []
        seen_majors: set[str] = set()

        if use_gliner:
            for match in self._predict_entities_chunked(text):
                label = match['label']

                if label in self._SKILL_LABELS:
                    skill_text = match['text'].lower().strip()
                    skill = self.skills_map.get(skill_text, skill_text)
                    if skill not in seen:
                        seen.add(skill)
                        skills[skill] = [match['start'], match['end']]

                elif label == "education degree":
                    normalized = normalize_degree(match['text'])
                    if normalized not in self._CANONICAL_DEGREES:
                        continue  # school name or other noise — skip

                    # Strip the degree token to recover the major from the span.
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

                    floor = self._DEGREE_CONFIDENCE_FLOOR[normalized]
                    if match['score'] >= floor:
                        degree_candidates.append((self._DEGREE_ORDER[normalized], match['score'], normalized))

                elif label == "education major":
                    major_lower = match['text'].lower().strip()
                    if major_lower and major_lower not in seen_majors:
                        majors.append(major_lower)
                        seen_majors.add(major_lower)

        education_level = max(degree_candidates, key=lambda x: (x[0], x[1]))[2] if degree_candidates else None
        education = {"degree": education_level, "majors": majors}

        return skills, education

    def extract_skills(self, text: str, use_gliner: bool = True) -> dict[str, list[int]]:
        """Extract skills from text. Wraps extract_all for backwards compatibility."""
        skills, _ = self.extract_all(text, use_gliner=use_gliner, use_spacy=True)
        return skills

    def extract_education(self, text: str) -> dict:
        """Extract education (degree + majors) from text. Wraps extract_all for backwards compatibility."""
        _, education = self.extract_all(text, use_gliner=True, use_spacy=False)
        return education
