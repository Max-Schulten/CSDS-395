from utils.resume_utils import ResumeClassifier
from utils.skills_utils import SkillsExtractor, load_gliner
from utils.gen_utils import load_classifier

class Resume:
    def __init__(self, resume_text, classifier: ResumeClassifier|None = None, skill_extractor: SkillsExtractor|None = None, gliner=None):
        classifier = classifier if classifier is not None else load_classifier()
        skill_extractor = skill_extractor if skill_extractor is not None else SkillsExtractor()
        self.resume_raw = resume_text
        self.education = skill_extractor.extract_education(resume_text)
        self.skills = skill_extractor.extract_skills(resume_text)
        self.resume_text = classifier.clean_resume(resume_text, gliner=gliner) # type: ignore

        
    def to_dict(self):
        return {
            "text": self.resume_text,
            "skills": self.skills,
            "education": self.education
        }