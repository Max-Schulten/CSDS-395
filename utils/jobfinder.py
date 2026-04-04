from utils.job import Job
from utils.resume import Resume
from utils.resume_utils import ResumeClassifier
from utils.skills_utils import SkillsExtractor
      
def find_jobs(resume: Resume, classifier: ResumeClassifier, skill_extractor: SkillsExtractor) -> list[Job]:
    return [Job("test", classifier=classifier, skill_extractor=skill_extractor)]