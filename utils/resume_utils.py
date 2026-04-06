import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from utils.gen_utils import load_classifier, load_nlp


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper for SentenceTransformer. Defined here so
    joblib can deserialize the saved (embedder, clf) tuple at load time."""
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(list(X), batch_size=self.batch_size)

    def encode(self, text, **kwargs):
        return self.model.encode(text, **kwargs)

class ResumeClassifier:
    """Classifies resumes into broad job categories after stripping PII.

    Attributes:
        model: Fitted sklearn pipeline.
        nlp: Loaded spaCy language model.
        embedder: SentenceTransformer model for text embedding.
        pii_patterns (dict[str, re.Pattern]): Compiled regex patterns for PII redaction.
        entities (list[str]): spaCy NER labels to redact.
    """

    def __init__(
        self,
        model = None,
        nlp_model = None,
        pii_entities: list[str] = ["PERSON", "GPE", "LOC", "ORG"]
    ) -> None:
        """Initialise classifier, loading the sklearn model, spaCy, and embedding model.

        Args:
            model: Tuple of (SentenceTransformerVectorizer, LinearSVC) loaded from joblib.
            nlp_model: Loaded spaCy language model.
            pii_entities (list[str]): NER labels whose matches will be redacted.

        Raises:
            FileNotFoundError: If no model file exists at model_dir.
            OSError: If the requested spaCy model package is not installed.
            AssertionError: If any value in entities is not a valid NER label
                for the loaded spaCy model.
        """
        self.embedder, self.model = model if model is not None else load_classifier()
        self.nlp = nlp_model if nlp_model is not None else load_nlp()

        self.pii_patterns = {
            "ADDRESS": re.compile(
                r'\b\d{1,3}(?!\d)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}'
                r'(?:\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard'
                r'|Dr|Drive|Ln|Lane|Way|Ct|Court|Pl|Place|Pkwy|Parkway'
                r'|Hwy|Highway|Cir|Circle|Terrace|Ter)\.?)?'
                r'(?:[,\s]+(?:Apt|Apartment|Suite|Ste|Unit|Fl|Floor|#)'
                r'\s*[\w-]+)?',
                re.IGNORECASE
            ),
            "PHONE": re.compile(
                r'(?:'
                r'(?:\+|00)\d{1,3}[\s\-.]?(?:\(?\d{1,4}\)?[\s\-.]?)?\d{1,4}[\s\-.]?\d{1,4}[\s\-.]?\d{1,9}'
                r'|'
                r'\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}'
                r'|'
                r'\d{5}[\s\-.]?\d{3}[\s\-.]?\d{2,3}'
                r'|'
                r'/(?:\+{0,1}\d+)(?:\s+|-)\({0,1}\d{1,3}\({0,1}(?:(?:\s*|-|\()\d{2,5}(?:\){0,1})\s*?){1,3}/'
                r')',
                re.IGNORECASE
            ),
            "ZIP":   re.compile(r'\b\d{5}(?:-\d{4})?\b|\b\d{6}\b'),
            "POBOX": re.compile(r'\bP\.?\s*O\.?\s*Box\s+\d+\b', re.IGNORECASE),
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
        }

        valid_ner_labels = self.nlp.get_pipe("ner").labels #type: ignore
        invalid = [e for e in pii_entities if e not in valid_ner_labels]
        assert not invalid, f"Invalid NER labels for '{nlp_model}': {invalid}"
        self.entities = pii_entities

    def clean_resume(self, resume: str, gliner = None) -> str:
        """Strip PII from a resume string using regex and spaCy NER.

        Regex redaction runs before NER so that character offsets
        produced by spaCy remain valid against the already-substituted text.

        Args:
            resume (str): Raw resume text.

        Returns:
            str: Resume text with PII replaced by bracketed tokens
                e.g. [EMAIL], [PHONE], [PERSON].

        Raises:
            TypeError: If resume is not a string.
            ValueError: If resume is empty.
        """
        if not isinstance(resume, str):
            raise TypeError(f"Expected str, got {type(resume).__name__}.")
        if not resume.strip():
            raise ValueError("Resume text is empty.")

        text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\/\(\)\@\+]', '', resume)
        text = re.sub(r'\s+', ' ', text).strip()

        for label, pattern in self.pii_patterns.items():
            text = pattern.sub(f'[{label}]', text)

        doc = self.nlp(text)
        for ent in reversed(doc.ents):
            if ent.label_ in self.entities:
                text = text[:ent.start_char] + f"[{ent.label_}]" + text[ent.end_char:]

        if gliner is not None:
            pii_labels = ["person name", "school"]
            other_labels = ["education degree", "education major"] # Disambiguates between false positive and true positive
            label_map = {"person name": "PERSON", "school": "ORG"}
            gliner_ents = gliner.predict_entities(text, pii_labels + other_labels , threshold=0.5)
            for ent in sorted(gliner_ents, key=lambda e: e['start'], reverse=True):
                label = ent['label']
                if label not in pii_labels:
                    continue
                label = label_map[label]
                text = text[:ent['start']] + f'[{label}]' + text[ent['end']:]

        return text

    def classify_resume(self, resume: str, top_k: int = 2) -> list[str]:
        """Classify a resume into the top-k most probable job categories.

        Args:
            resume (str): Raw resume text. PII removal is applied internally.
            top_k (int): Number of categories to return, ordered by descending
                probability. Must be between 1 and the number of known classes.

        Returns:
            list[str]: Top-k category labels, e.g. ['Technology', 'Engineering'].
                One of: Technology, Business & Management, Engineering,
                Human & Social Services, Sales & Marketing, Finance & Accounting,
                Creative & Design, Health & Lifestyle, Building & Construction,
                Aviation & Transport, Automobile, Agriculture.

        Raises:
            ValueError: If top_k is less than 1 or exceeds the number of classes.
        """
        n_classes = len(self.model.classes_)
        if not 1 <= top_k <= n_classes:
            raise ValueError(f"top_k must be between 1 and {n_classes}, got {top_k}.")

        cleaned = self.clean_resume(resume)
        embedded = self.embedder.encode(cleaned).reshape(1, -1)
        proba = self.model.decision_function(embedded)[0]
        topk_indices = np.argsort(proba)[::-1][:top_k]
        return [self.model.classes_[i] for i in topk_indices]