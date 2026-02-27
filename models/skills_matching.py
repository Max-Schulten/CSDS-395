import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
import json
from config import BASE_DIR
import os

if __name__ == "__main__":
    resume = """
    Experienced project manager using Excel, and Google Drive
    """

    job = """
    We are looking for a backend engineer proficient in Python and Flask.
    Experience with Docker, Kubernetes, and AWS required.
    PostgreSQL and Redis experience preferred. Familiarity with CI/CD a plus.
    """

    skills_df = pd.read_csv(os.path.join(BASE_DIR,"data/skills_en.csv"))
    onet_skills = set(pd.read_csv(os.path.join(BASE_DIR,"data/onet/skills.txt"), sep='\t')['Element Name'].str.lower().str.strip().tolist())
    onet_knowledge = set(pd.read_csv(os.path.join(BASE_DIR,"data/onet/knowledge.txt"), sep='\t')['Element Name'].str.lower().str.strip().tolist())
    onet_tech = set(pd.read_csv(os.path.join(BASE_DIR, "data/onet/tech.txt"), sep='\t')['Example'].str.lower().str.strip().tolist())
    nlp = spacy.load("en_core_web_md", disable=["parser"])
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    pattern_to_preferred = {}
    patterns = set()
    for _, row in skills_df.iterrows():
        preferred = row["preferredLabel"].strip().lower()
        patterns.add(nlp.make_doc(preferred))
        pattern_to_preferred[preferred.lower()] = preferred
        if pd.isna(row['altLabels']):
            continue
        for alt in row["altLabels"].split("\n"):
            if alt.strip():
                pattern_to_preferred[alt.strip().lower()] = preferred
                patterns.add(nlp.make_doc(alt.strip().lower()))
    for skill in onet_skills:
        pattern_to_preferred.setdefault(skill, skill)
        patterns.add(nlp.make_doc(skill))
    for skill in onet_knowledge:
        pattern_to_preferred.setdefault(skill, skill)
        patterns.add(nlp.make_doc(skill))
    for skill in onet_tech:
        deorged_skill = skill
        for ent in reversed(nlp(skill).ents):
            if ent.label_ == 'ORG':
                deorged_skill = deorged_skill[:ent.start_char] + deorged_skill[ent.end_char:]
                if deorged_skill and len(deorged_skill) > 0 and deorged_skill != skill:
                    pattern_to_preferred.setdefault(skill, deorged_skill)
                    patterns.add(nlp.make_doc(deorged_skill))
                    
        pattern_to_preferred.setdefault(skill, skill)
        patterns.add(nlp.make_doc(skill))
    with open(os.path.join(BASE_DIR,"data/skill_map.json"), "w") as f:
        json.dump(pattern_to_preferred, f)
    