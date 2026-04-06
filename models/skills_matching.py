import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import spacy
import pandas as pd
import json
from config import BASE_DIR
INCLUDE_DEORGED = False

if __name__ == "__main__":
    resume = """
    Experienced project manager using Excel, and Google Drive
    """

    job = """
    We are looking for a backend engineer proficient in Python and Flask.
    Experience with Docker, Kubernetes, and AWS required.
    PostgreSQL and Redis experience preferred. Familiarity with CI/CD a plus.
    """

    skills_df = pd.read_csv(os.path.join(BASE_DIR, "data/skills_en.csv"))
    onet_tech = set(pd.read_csv(os.path.join(BASE_DIR, "data/onet/tech.txt"), sep='\t')['Example'].str.lower().str.strip().tolist())
    nlp = spacy.load("en_core_web_md", disable=["parser"])

    pattern_to_preferred = {}
    pattern_strings: set[str] = set()  # deduplicate on string level before making Docs

    # ESCO - preferred labels take priority; alt labels map to preferred
    for _, row in skills_df.iterrows():
        preferred = row["preferredLabel"].strip().lower()
        pattern_to_preferred[preferred] = preferred
        pattern_strings.add(preferred)
        if pd.isna(row['altLabels']):
            continue
        for alt in row["altLabels"].split("\n"):
            alt = alt.strip().lower()
            if alt:
                pattern_to_preferred.setdefault(alt, preferred)
                pattern_strings.add(alt)

    # O*NET tech - strip ORG entities (e.g. "Microsoft Excel" -> "Excel")
    # Entities are processed right-to-left so earlier offsets stay valid.
    if INCLUDE_DEORGED:
        for skill in onet_tech:
            deorged = skill
            for ent in reversed(nlp(skill).ents):
                if ent.label_ == 'ORG':
                    deorged = (deorged[:ent.start_char] + deorged[ent.end_char:]).strip()

            if deorged and deorged != skill and len(deorged) > 1 and not nlp.vocab[deorged].is_stop:
                # Original form maps to the stripped preferred label
                pattern_to_preferred.setdefault(skill, deorged)
                # Stripped form maps to itself (so it's also matchable directly)
                pattern_to_preferred.setdefault(deorged, deorged)
                pattern_strings.add(deorged)

            pattern_to_preferred.setdefault(skill, skill)
            pattern_strings.add(skill)

    with open(os.path.join(BASE_DIR, "data/skill_map.json"), "w") as f:
        json.dump(pattern_to_preferred, f)
