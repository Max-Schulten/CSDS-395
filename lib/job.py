from lib.skills_utils import SkillsExtractor
from lib.gen_utils import clean_text

class Job:
    def __init__(self, job_desc, skill_extractor: SkillsExtractor):
        skill_extractor = skill_extractor if skill_extractor is not None else SkillsExtractor()
        self.job_desc = clean_text(job_desc)
        self.skills, self.education = skill_extractor.extract_all(self.job_desc, use_gliner=True, use_spacy=False)
    
    def to_dict(self):
        return {
            "job_desc": self.job_desc,
            "skills": self.skills,
            "education": self.education
        }