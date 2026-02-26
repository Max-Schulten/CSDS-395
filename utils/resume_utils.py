import joblib
import os
import spacy, spacy.util
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from huggingface_hub import model_info
from huggingface_hub.errors import RepositoryNotFoundError


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
        model_dir: str = "../models/resume_classifier.joblib",
        nlp_model: str = "en_core_web_md",
        embedding_model: str = "all-MiniLM-L6-v2",
        entities: list[str] = ["PERSON", "GPE", "LOC"],
    ) -> None:
        """Initialise classifier, loading the sklearn model, spaCy, and embedding model.

        Args:
            model_dir (str): Path to a saved joblib model file.
            nlp_model (str): Name of an installed spaCy model package.
            embedding_model (str): HuggingFace model name for SentenceTransformer.
            entities (list[str]): NER labels whose matches will be redacted.

        Raises:
            FileNotFoundError: If no model file exists at model_dir.
            OSError: If the requested spaCy model package is not installed.
            ValueError: If embedding_model is not a valid HuggingFace model.
            AssertionError: If any value in entities is not a valid NER label
                for the loaded spaCy model.
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No model found at '{model_dir}'.")
        self.model = joblib.load(model_dir)

        if not spacy.util.is_package(nlp_model):
            raise OSError(f"spaCy model '{nlp_model}' is not installed.")
        self.nlp = spacy.load(nlp_model)

        try:
            model_info(f"sentence-transformers/{embedding_model}")
        except RepositoryNotFoundError:
            raise ValueError(f"'{embedding_model}' is not a valid SentenceTransformer model.")
        self.embedder = SentenceTransformer(embedding_model)

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

        valid_ner_labels = self.nlp.get_pipe("ner").labels
        invalid = [e for e in entities if e not in valid_ner_labels]
        assert not invalid, f"Invalid NER labels for '{nlp_model}': {invalid}"
        self.entities = entities

    def clean_resume(self, resume: str) -> str:
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