#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:29:09 2026
@author: maximilianschulten
"""
import pandas as pd
import spacy
import re

SPACY_MODEL = "en_core_web_md"

df = pd.read_csv("./data/resume-class.csv") # Source: https://arxiv.org/pdf/2406.18125

"""
Dataset categories are too granular.
We map them onto slightly broader categories.
Epsecially applicable for developer roles.
"""
CATEGORY_MAP = {
    # Engineering
    "Electrical Engineering":       "Engineering",
    "Mechanical Engineer":          "Engineering",
    "Civil Engineer":               "Engineering",
    "Architecture":                 "Engineering",
    # Technology
    "Java Developer":               "Technology",
    "Python Developer":             "Technology",
    "React Developer":              "Technology",
    "DotNet Developer":             "Technology",
    "SAP Developer":                "Technology",
    "ETL Developer":                "Technology",
    "SQL Developer":                "Technology",
    "DevOps":                       "Technology",
    "Database":                     "Technology",
    "Network Security Engineer":    "Technology",
    "Information Technology":       "Technology",
    "Web Designing":                "Technology",
    "Blockchain":                   "Technology",
    "Data Science":                 "Technology",
    "Testing":                      "Technology",
    # Business & Management
    "Management":                   "Business & Management",
    "Consultant":                   "Business & Management",
    "Business Analyst":             "Business & Management",
    "Operations Manager":           "Business & Management",
    "PMO":                          "Business & Management",
    "BPO":                          "Business & Management",
    # Finance & Accounting
    "Accountant":                   "Finance & Accounting",
    "Finance":                      "Finance & Accounting",
    "Banking":                      "Finance & Accounting",
    # Sales & Marketing
    "Sales":                        "Sales & Marketing",
    "Public Relations":             "Sales & Marketing",
    "Digital Media":                "Sales & Marketing",
    # Human & Social Services
    "Human Resources":              "Human & Social Services",
    "Education":                    "Human & Social Services",
    "Advocate":                     "Human & Social Services",
    # Creative & Design
    "Arts":                         "Creative & Design",
    "Designing":                    "Creative & Design",
    "Apparel":                      "Creative & Design",
    # Health & Lifestyle
    "Health and Fitness":           "Health & Lifestyle",
    "Food and Beverages":           "Health & Lifestyle",
    # Aviation, Transport, Automobile, Construction
    "Automobile":                   "Automobile",
    "Aviation":                     "Aviation & Transport",
    "Agriculture":                  "Agriculture",
    "Building and Construction":    "Building & Construction",
}

df['Mapped_Category'] = df['Category'].map(CATEGORY_MAP)

nlp = spacy.load(SPACY_MODEL)
entities = ["PERSON", "GPE", "LOC"]

ADDRESS_RE = re.compile(
    r'\b\d{1,3}(?!\d)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}'
    r'(?:\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard'
    r'|Dr|Drive|Ln|Lane|Way|Ct|Court|Pl|Place|Pkwy|Parkway'
    r'|Hwy|Highway|Cir|Circle|Terrace|Ter)\.?)?'
    r'(?:[,\s]+(?:Apt|Apartment|Suite|Ste|Unit|Fl|Floor|#)'
    r'\s*[\w-]+)?',
    re.IGNORECASE
)

ZIP_RE = re.compile(r'\b\d{5}(?:-\d{4})?\b|\b\d{6}\b')

POBOX_RE = re.compile(r'\bP\.?\s*O\.?\s*Box\s+\d+\b', re.IGNORECASE)

EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
)

PHONE_RE = re.compile(
    r'(?:'
    r'(?:\+|00)\d{1,3}[\s\-.]?(?:\(?\d{1,4}\)?[\s\-.]?)?\d{1,4}[\s\-.]?\d{1,4}[\s\-.]?\d{1,9}'
    r'|'
    r'\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}'
    r'|'
    r'\d{5}[\s\-.]?\d{3}[\s\-.]?\d{2,3}'
    r')',
    re.IGNORECASE
)


def clean(texts, nlp, entities, NER=True):
    cleaned = []
    for text in texts:
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\/\(\)\@\+]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        text = EMAIL_RE.sub('[EMAIL]', text)
        text = PHONE_RE.sub('[PHONE]', text)
        text = POBOX_RE.sub('[ADDRESS]', text)
        text = ADDRESS_RE.sub('[ADDRESS]', text)
        text = ZIP_RE.sub('[ZIP]', text)
        
        if not NER:
            return text
        doc = nlp(text)
        new_text = text
        for ent in reversed(doc.ents):
            if ent.label_ in entities:
                new_text = new_text[:ent.start_char] + f"[{ent.label_}]" + new_text[ent.end_char:]

        cleaned.append(new_text)
    return cleaned


df['Text'] = clean(df['Text'], nlp, entities)

df.to_csv(f"./data/dataset-{SPACY_MODEL}.csv", index=False)