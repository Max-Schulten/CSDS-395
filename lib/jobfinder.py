from lib.job import Job
from lib.resume import Resume
from lib.resume_utils import ResumeClassifier
from lib.skills_utils import SkillsExtractor
      
def find_jobs(resume: Resume, skill_extractor: SkillsExtractor) -> list[Job]:
    return [Job("test", skill_extractor=skill_extractor)]