from lib.resume_utils import ResumeClassifier
from lib.skills_utils import SkillsExtractor
from lib.gen_utils import load_classifier

class Resume:
    def __init__(self, resume_text, classifier: ResumeClassifier|None = None, skill_extractor: SkillsExtractor|None = None, gliner=None):
        classifier = classifier if classifier is not None else load_classifier()
        skill_extractor = skill_extractor if skill_extractor is not None else SkillsExtractor()
        self.resume_raw = resume_text
        self.skills, self.education = skill_extractor.extract_all(resume_text, use_gliner=True, use_spacy=True)
        self.resume_text = classifier.clean_resume(resume_text, gliner=gliner) # type: ignore
        self.categories = classifier.classify_resume(self.resume_text) # type: ignore

    def to_dict(self):
        return {
            "text": self.resume_text,
            "skills": self.skills,
            "education": self.education,
            "categories": self.categories
        }