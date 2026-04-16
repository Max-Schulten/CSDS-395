#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:45:03 2026
@author: maximilianschulten
This script is used to create, test, train, and evaluate a classifier
that maps plain text résumés to a set of given categories.
"""
#%% IMPORTS + CONFIG
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import re
import glob
import spacy
import joblib
from lib.resume_utils import SentenceTransformerVectorizer

RS = 420
SYNTH_INCLUDED = False
SYNTH_MODEL = "Qwen7B"
SPACY_MODEL = "en_core_web_md"

#%% DATA
data = pd.read_csv(f"../data/dataset-{SPACY_MODEL}.csv")
print(f"# of examples: {data.shape[0]}")
print("Label Distribution")
print(data['Mapped_Category'].value_counts())
X = data['Text']
y = data['Mapped_Category']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RS)
print(f"# of training samples: {X_tr.shape[0]}")
print(f"# of test samples: {X_te.shape[0]}")
print(
    """
    Since classes are imbalanced, we stratify the holdout set.
    Overall, due to the imbalanced nature of the dataset, we
    evaluate our models on Macro & Micro f1, per-class 
    precision, recall, and f1. Confusion matrices are used to
    evaluate where errors are being made.
    """
    )
print("Training label distribution:")
print(y_tr.value_counts())
print("Testing label distribution:")
print(y_te.value_counts())

if SYNTH_INCLUDED:
    def strip_label_header(text, category):
        """Remove first line if it leaks the category label (common in small-model outputs)."""
        lines = text.splitlines()
        for i, line in enumerate(lines):
            stripped = re.sub(r"[*_`#]", "", line).strip()
            if stripped and category.lower() in stripped.lower():
                lines[i] = ""
                break
            elif stripped:
                break  # first non-empty line doesn't contain label — leave as-is
        return "\n".join(lines).lstrip()

    synth_files = glob.glob(f"../data/synthetic_resumes/{SYNTH_MODEL}/*.csv")
    synth_df = pd.concat([pd.read_csv(f) for f in synth_files], ignore_index=True)
    synth_df['Resume_str'] = synth_df.apply(
        lambda row: strip_label_header(row['Resume_str'], row['Category']), axis=1
    )
    X_tr = pd.concat([X_tr, synth_df['Resume_str']], ignore_index=True)
    y_tr = pd.concat([y_tr, synth_df['Category']], ignore_index=True)
    print(f"Synthetic data included: {len(synth_df)} additional training samples")
    print("Training label distribution after augmentation:")
    print(y_tr.value_counts())

#%% TUNING
class SpacyNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        docs = self.nlp.pipe(X, disable=["parser", "ner"])
        return [
            " ".join([token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha])
            for doc in docs
        ]

def simple_normalize(docs):
    cleaned = []
    for doc in docs:
        text = doc.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned

def top_k_accuracy(pipeline, X, y_true, k=3):
    decision_scores = pipeline.decision_function(X)
    classes = pipeline.classes_
    correct = 0
    for i, true_label in enumerate(y_true):
        top_k_labels = classes[np.argsort(decision_scores[i])[::-1][:k]]
        if true_label in top_k_labels:
            correct += 1
    return correct / len(y_true)

SimpleNormalizer = FunctionTransformer(simple_normalize)

embedder = SentenceTransformerVectorizer()
print("Embedding...")
X_tr_input = embedder.transform(X_tr)
X_te_input = embedder.transform(X_te)
print("Embedding Complete")

#%% 
pipe = Pipeline([
    ("clf", LinearSVC(max_iter=3000, random_state=10))
])
param_grid = {
    "clf__C": [0.4, 1, 2, 5, 10],
    "clf__class_weight": ['balanced', None],
    "clf__penalty": ['l2']
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RS
)

gs = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

gs.fit(X_tr_input, y_tr)

print("Best params:")
print(gs.best_params_)

print("Mean scores:")
print(gs.cv_results_['mean_test_score'])

best_pipe = gs.best_estimator_

y_pred_te = best_pipe.predict(X_te_input)

y_pred_tr = best_pipe.predict(X_tr_input)
print("TRAIN")
print(classification_report(y_tr, y_pred_tr))

print("TEST")
print(classification_report(y_te, y_pred_te))
ks = [1,2,3,4,5]
for k in ks:
    print(f"Top test {k} accuracy:")
    print(top_k_accuracy(best_pipe, X_te_input, y_te, k=k))


#%% FINAL MODEL — retrain on full dataset with best params
best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
final_clf = LinearSVC(max_iter=3000, random_state=RS, **best_params)
X_all_input = embedder.transform(pd.concat([X_tr, X_te]))
final_clf.fit(X_all_input, pd.concat([y_tr, y_te]))
joblib.dump((embedder, final_clf), "../models/resume_classifier.joblib")