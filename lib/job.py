from lib.skills_utils import SkillsExtractor
from lib.gen_utils import clean_text

class Job:
    def __init__(self, job_desc, skill_extractor: SkillsExtractor, job_title=None, company=None, loc=None, url=None):
        skill_extractor = skill_extractor if skill_extractor is not None else SkillsExtractor()
        self.job_title = clean_text(job_title) if job_title is not None else None
        self.job_desc = clean_text(job_desc)
        text = self.job_desc if self.job_title is None else self.job_title + "\n" + self.job_desc
        self.skills, self.education = skill_extractor.extract_all(text, use_gliner=True, use_spacy=True)
        self.company = company; self.loc=loc
        self.url = url
    def to_dict(self):
        return {
            "job_title": self.job_title,
            "job_desc": self.job_desc,
            "skills": self.skills,
            "education": self.education,
            "company": self.company,
            "location": self.loc,
            "url": self.url
        }