from utils.skills_utils import SkillsExtractor
from utils.gen_utils import clean_text

class Job:
    def __init__(self, job_desc, skill_extractor: SkillsExtractor):
        skill_extractor = skill_extractor if skill_extractor is not None else SkillsExtractor()
        self.job_desc = clean_text(job_desc)
        self.skills = skill_extractor.extract_skills(self.job_desc)
        self.education = skill_extractor.extract_education(self.job_desc)
    
    def to_dict(self):
        return {
            "job_desc": self.job_desc,
            "skills": self.skills,
            "education": self.education
        }