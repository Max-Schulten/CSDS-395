#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:45:03 2026
@author: maximilianschulten
This script is used to create, test, train, and evaluate a classifier
that maps plain text résumés to a set of given categories.
"""
#%% IMPORTS + CONFIG
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
from sentence_transformers import SentenceTransformer
import joblib

RS = 420
SYNTH_INCLUDED = False
EMBED = True
SPACY_MODEL = "en_core_web_md"

#%% DATA
data = pd.read_csv(f"./data/dataset-{SPACY_MODEL}.csv")
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
    synth_files = glob.glob("./data/synthetic_resumes/*.csv")
    synth_df = pd.concat([pd.read_csv(f) for f in synth_files], ignore_index=True)
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


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(
            list(X),
            batch_size=self.batch_size
        )

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

if EMBED:
    embedder = SentenceTransformerVectorizer()
    print("Embedding...")
    X_tr_input = embedder.transform(X_tr)
    X_te_input = embedder.transform(X_te)
    print("Embedding Complete")
    pipe = Pipeline([
        ("clf", LinearSVC(max_iter=3000, random_state=RS)),
    ])
    param_grid = {
        "clf__C": [0.2, 0.3, 0.4],
        "clf__class_weight": ['balanced', None],
        "clf__penalty": ['l1', 'l2']
    }
else:
    X_tr_input = X_tr
    X_te_input = X_te
    pipe = Pipeline([
        ("normalize", SimpleNormalizer),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            max_features=5_000
        )),
        ("clf", LinearSVC(class_weight='balanced', max_iter=3000, random_state=RS, C=0.5)),
    ])
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": [2, 3, 5],
        "tfidf__max_features": [3_000, 5_000, 10_000],
        "clf__C": [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
    }

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RS
)

gs = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="f1_macro",
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

#%% 
joblib.dump(gs.best_estimator_, "./models/resume_classifier.joblib")