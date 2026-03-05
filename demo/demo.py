#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:58:14 2026

@author: maximilianschulten
"""
#%% SETUP
from utils.gen_utils import load_embedder, load_classifier, load_nlp, clean_text
from config import BASE_DIR
from utils.skills_utils import load_gliner
from utils.resume_utils import ResumeClassifier
from utils.skills_utils import SkillsExtractor, load_skills_matcher, load_skills_map
import os 
print("Loading Skills Map...")
skills_map = load_skills_map(os.path.join(BASE_DIR, "data/skill_map.json"))
print("Skills map loaded.")

print("Loading Models...")
nlp = load_nlp()
print("NLP Loaded")
embedding_model = load_embedder()
print("Embedder loaded")
gliner = load_gliner()
print("GliNER loaded")
classifier = ResumeClassifier(model=load_classifier(), nlp_model=nlp, embedding_model=embedding_model)
print("SVC Loaded")
matcher = load_skills_matcher(nlp_model=nlp)
print("Skills Matcher Loaded")
skill_extractor = SkillsExtractor(nlp=nlp, matcher=matcher, gliner=gliner, skills_map=skills_map)
print("Skills Extractor loaded")
print("All Models loaded.")

DUMMY_RESUME = """ALEX RIVERS
(555) 012-3456 | alex.rivers.dev@email.com
linkedin.com/in/arivers-dev | github.com/arivers-codes | San Francisco, CA

SUMMARY
Detail-oriented Software Engineer with 5 years of experience building 
scalable web applications and optimizing backend systems. Proven track record 
of collaborating in agile environments to deliver high-quality code and 
improving system performance by 35%.

SKILLS
Languages:      Python, Java, TypeScript
Frameworks:     React, Node.js, Spring Boot
Tools/Cloud:    Docker, Kubernetes, AWS (EC2, S3, Lambda)
Databases:      PostgreSQL, MongoDB, Redis

PROFESSIONAL EXPERIENCE
TechFlow Solutions | Senior Software Engineer
Remote | January 2021 – Present
* Designed and implemented a real-time analytics dashboard using React and 
    Node.js, resulting in a 20% increase in user engagement.
* Optimized database queries and API endpoints, reducing average response 
    latency by 150ms.
* Led a team of 4 engineers in the migration of a legacy monolithic billing 
    service to a microservices architecture.
* Automated CI/CD pipelines using GitHub Actions, reducing deployment time 
    from 45 to 12 minutes.

CloudScale Inc. | Software Engineer
San Jose, CA | June 2018 – December 2020
* Developed a high-throughput data processing module which streamlined 
    inventory management for 500+ internal users.
* Resolved over 120 critical production bugs, maintaining a 99.9% 
    system uptime during peak traffic periods.
* Collaborated with cross-functional teams (Product, Design, QA) 
    to define technical requirements for three major product launches.

PROJECTS
OpenSource Task Manager | github.com/arivers-codes/task-pro
* Built a collaborative task management tool using GraphQL and Apollo to 
    solve synchronization delays in remote teams.
* Implemented a custom caching layer which handled over 10,000 concurrent 
    websocket connections.

EDUCATION
State University of Technology | B.S. in Computer Science
Graduated May 2018
"""
#%% Classifier Demo
classifier = ResumeClassifier(model=load_classifier(), nlp_model=nlp, embedding_model=embedding_model)
cleaned_resume = classifier.clean_resume(DUMMY_RESUME)
print('-'*50)
print('Raw Resume:')
print(DUMMY_RESUME)
print('-'*50)
print('Cleaned Resume:')
print(cleaned_resume)
print('-'*50)
classes = classifier.classify_resume(cleaned_resume, top_k=2)
print('Top 2 Most Likely Classes')
print(classes)
print('-'*50)
"""
Pipeline:
    Resume -> cleaning -> PII removal -> embedding model creates dense vector representation of text -> SVM w/ L2 regularization
"""
    

#%% Skill extractor Demo
print('-'*50)
print('Extracted Skills')
print(skill_extractor.extract_skills(cleaned_resume))

"""
Pipeline:
    Clean text -> NER Model -> Mapped to O*NET/ESCO skill -> GliNER performs one shot NER -> Text Spans and labels extracted
"""
