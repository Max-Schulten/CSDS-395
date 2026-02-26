# tests/test_resume_classifier.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from utils import ResumeClassifier


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_model():
    mock_model = MagicMock()
    mock_model.classes_ = np.array([
        "Technology", "Engineering", "Finance & Accounting",
        "Business & Management", "Human & Social Services",
        "Sales & Marketing", "Creative & Design", "Health & Lifestyle",
        "Building and Construction", "Aviation & Transport",
        "Automobile", "Agriculture"
    ])
    mock_model.predict_proba.return_value = np.array([[
        0.6, 0.2, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01
    ]])
    return mock_model


def make_mock_nlp():
    mock_nlp = MagicMock()
    mock_nlp.get_pipe.return_value.labels = ("PERSON", "ORG", "GPE", "LOC")
    mock_nlp.return_value.ents = []
    return mock_nlp


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classifier():
    """ResumeClassifier with all external dependencies mocked."""
    with patch("utils.resume_utils.os.path.exists", return_value=True), \
         patch("utils.resume_utils.joblib.load", return_value=make_mock_model()), \
         patch("utils.resume_utils.spacy.util.is_package", return_value=True), \
         patch("utils.resume_utils.spacy.load", return_value=make_mock_nlp()), \
         patch("utils.resume_utils.model_info"), \
         patch("utils.resume_utils.SentenceTransformer") as mock_st:

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros(384)
        mock_st.return_value = mock_embedder

        return ResumeClassifier()


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_raises_if_model_not_found(self):
        with patch("utils.resume_utils.os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                ResumeClassifier()

    def test_raises_if_spacy_model_not_installed(self):
        with patch("utils.resume_utils.os.path.exists", return_value=True), \
             patch("utils.resume_utils.joblib.load"), \
             patch("utils.resume_utils.spacy.util.is_package", return_value=False):
            with pytest.raises(OSError):
                ResumeClassifier()

    def test_raises_on_invalid_embedding_model(self):
        from huggingface_hub.errors import RepositoryNotFoundError
        with patch("utils.resume_utils.os.path.exists", return_value=True), \
             patch("utils.resume_utils.joblib.load"), \
             patch("utils.resume_utils.spacy.util.is_package", return_value=True), \
             patch("utils.resume_utils.spacy.load", return_value=make_mock_nlp()), \
             patch("utils.resume_utils.model_info", side_effect=RepositoryNotFoundError("")):
            with pytest.raises(ValueError):
                ResumeClassifier()

    def test_raises_on_invalid_ner_label(self):
        with patch("utils.resume_utils.os.path.exists", return_value=True), \
             patch("utils.resume_utils.joblib.load"), \
             patch("utils.resume_utils.spacy.util.is_package", return_value=True), \
             patch("utils.resume_utils.spacy.load") as mock_spacy, \
             patch("utils.resume_utils.model_info"), \
             patch("utils.resume_utils.SentenceTransformer"):
            mock_nlp = MagicMock()
            mock_nlp.get_pipe.return_value.labels = ("PERSON", "GPE")
            mock_spacy.return_value = mock_nlp
            with pytest.raises(AssertionError):
                ResumeClassifier(entities=["PERSON", "INVALID"])


# ── clean_resume ──────────────────────────────────────────────────────────────

class TestCleanResume:
    def test_raises_on_non_string(self, classifier):
        with pytest.raises(TypeError):
            classifier.clean_resume(123)

    def test_raises_on_empty_string(self, classifier):
        with pytest.raises(ValueError):
            classifier.clean_resume("   ")

    def test_email_redacted(self, classifier):
        result = classifier.clean_resume("Contact me at john.doe@gmail.com for more info.")
        assert "[EMAIL]" in result
        assert "john.doe@gmail.com" not in result

    def test_us_phone_redacted(self, classifier):
        result = classifier.clean_resume("Call me at (555) 123-4567.")
        assert "[PHONE]" in result
        assert "555" not in result

    def test_international_phone_redacted(self, classifier):
        result = classifier.clean_resume("Reach me at +91 98765 43210.")
        assert "[PHONE]" in result

    def test_street_address_redacted(self, classifier):
        result = classifier.clean_resume("I live at 123 Main Street, Apt 4B.")
        assert "[ADDRESS]" in result
        assert "123 Main Street" not in result

    def test_zip_redacted(self, classifier):
        result = classifier.clean_resume("My zip code is 90210.")
        assert "[ZIP]" in result
        assert "90210" not in result

    def test_pobox_redacted(self, classifier):
        result = classifier.clean_resume("Send mail to P.O. Box 1234.")
        assert "[POBOX]" in result

    def test_ner_entity_redacted(self, classifier):
        mock_ent = MagicMock()
        mock_ent.text = "John Smith"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 10
        classifier.nlp.return_value.ents = [mock_ent]
        result = classifier.clean_resume("John Smith is a software engineer.")
        assert "[PERSON]" in result

    def test_clean_text_unchanged(self, classifier):
        text = "Experienced software engineer with Python and machine learning skills."
        result = classifier.clean_resume(text)
        assert result == text


# ── classify_resume ───────────────────────────────────────────────────────────

class TestClassifyResume:
    def test_returns_top_k_categories(self, classifier):
        result = classifier.classify_resume("Python developer with 5 years experience.", top_k=2)
        assert len(result) == 2

    def test_top_1_is_highest_proba(self, classifier):
        result = classifier.classify_resume("Python developer with 5 years experience.", top_k=1)
        assert result[0] == "Technology"

    def test_raises_on_invalid_top_k_zero(self, classifier):
        with pytest.raises(ValueError):
            classifier.classify_resume("Some resume.", top_k=0)

    def test_raises_on_top_k_exceeds_classes(self, classifier):
        with pytest.raises(ValueError):
            classifier.classify_resume("Some resume.", top_k=999)

    def test_returns_list_of_strings(self, classifier):
        result = classifier.classify_resume("Some resume.", top_k=2)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_embedder_called_with_cleaned_text(self, classifier):
        classifier.classify_resume("Python developer at john@example.com.", top_k=1)
        call_arg = classifier.embedder.encode.call_args[0][0]
        assert "john@example.com" not in call_arg
        assert "[EMAIL]" in call_arg