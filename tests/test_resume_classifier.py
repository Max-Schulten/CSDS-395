# tests/test_resume_classifier.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from utils.resume_utils import ResumeClassifier


# ── Helpers ───────────────────────────────────────────────────────────────────

CLASSES = np.array([
    "Technology", "Engineering", "Finance & Accounting",
    "Business & Management", "Human & Social Services",
    "Sales & Marketing", "Creative & Design", "Health & Lifestyle",
    "Building and Construction", "Aviation & Transport",
    "Automobile", "Agriculture"
])

# decision_function scores — highest for "Technology"
DECISION_SCORES = np.array([
    2.5, 1.2, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05,
    0.03, 0.02, 0.01, 0.0
])


def make_mock_model():
    mock_model = MagicMock()
    mock_model.classes_ = CLASSES
    mock_model.decision_function.return_value = np.array([DECISION_SCORES])
    return mock_model


def make_mock_nlp():
    mock_nlp = MagicMock()
    mock_nlp.get_pipe.return_value.labels = ("PERSON", "ORG", "GPE", "LOC")
    mock_nlp.return_value.ents = []
    return mock_nlp


def make_mock_embedder():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(384)
    return mock_embedder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    return make_mock_model()


@pytest.fixture
def mock_nlp():
    return make_mock_nlp()


@pytest.fixture
def mock_embedder():
    return make_mock_embedder()


@pytest.fixture
def classifier(mock_model, mock_nlp, mock_embedder):
    """ResumeClassifier with all dependencies injected — no I/O."""
    return ResumeClassifier(
        model=mock_model,
        nlp_model=mock_nlp,
        embedding_model=mock_embedder,
    )


# ── gen_utils load functions ──────────────────────────────────────────────────
# Validation now lives in gen_utils, so test it there rather than via __init__.

class TestLoadClassifier:
    def test_raises_if_model_not_found(self):
        from utils.gen_utils import load_classifier
        with patch("utils.gen_utils.os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="No model found"):
                load_classifier()

    def test_returns_model_when_file_exists(self, mock_model):
        from utils.gen_utils import load_classifier
        with patch("utils.gen_utils.os.path.exists", return_value=True), \
             patch("utils.gen_utils.joblib.load", return_value=mock_model):
            result = load_classifier()
            assert result is mock_model


class TestLoadNlp:
    def test_raises_if_spacy_model_not_installed(self):
        from utils.gen_utils import load_nlp
        with patch("utils.gen_utils.spacy.util.is_package", return_value=False):
            with pytest.raises(OSError, match="not installed"):
                load_nlp()

    def test_returns_nlp_when_installed(self, mock_nlp):
        from utils.gen_utils import load_nlp
        with patch("utils.gen_utils.spacy.util.is_package", return_value=True), \
             patch("utils.gen_utils.spacy.load", return_value=mock_nlp):
            result = load_nlp()
            assert result is mock_nlp


class TestLoadEmbedder:
    def test_raises_on_invalid_model(self):
        from utils.gen_utils import load_embedder
        from huggingface_hub.errors import RepositoryNotFoundError
        with patch("utils.gen_utils.model_info", side_effect=RepositoryNotFoundError("")):
            with pytest.raises(ValueError, match="not a valid SentenceTransformer"):
                load_embedder(model="not-a-real-model")

    def test_returns_embedder_on_valid_model(self, mock_embedder):
        from utils.gen_utils import load_embedder
        with patch("utils.gen_utils.model_info"), \
             patch("utils.gen_utils.SentenceTransformer", return_value=mock_embedder):
            result = load_embedder()
            assert result is mock_embedder


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_stores_injected_dependencies(self, mock_model, mock_nlp, mock_embedder):
        rc = ResumeClassifier(
            model=mock_model,
            nlp_model=mock_nlp,
            embedding_model=mock_embedder,
        )
        assert rc.model is mock_model
        assert rc.nlp is mock_nlp
        assert rc.embedder is mock_embedder

    def test_raises_on_invalid_ner_label(self, mock_model, mock_nlp, mock_embedder):
        # mock_nlp only has PERSON, ORG, GPE, LOC
        with pytest.raises(AssertionError, match="Invalid NER labels"):
            ResumeClassifier(
                model=mock_model,
                nlp_model=mock_nlp,
                embedding_model=mock_embedder,
                pii_entities=["PERSON", "INVALID_LABEL"],
            )

    def test_valid_pii_entities_accepted(self, mock_model, mock_nlp, mock_embedder):
        rc = ResumeClassifier(
            model=mock_model,
            nlp_model=mock_nlp,
            embedding_model=mock_embedder,
            pii_entities=["PERSON", "GPE"],
        )
        assert rc.entities == ["PERSON", "GPE"]

    def test_default_pii_entities(self, classifier):
        assert classifier.entities == ["PERSON", "GPE", "LOC"]

    def test_pii_patterns_compiled(self, classifier):
        import re
        for label, pattern in classifier.pii_patterns.items():
            assert isinstance(pattern, re.Pattern), f"{label} pattern not compiled"


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

    def test_ner_entity_not_in_pii_list_kept(self, classifier):
        """ORG is not in default pii_entities so should not be redacted."""
        mock_ent = MagicMock()
        mock_ent.text = "OpenAI"
        mock_ent.label_ = "ORG"
        mock_ent.start_char = 0
        mock_ent.end_char = 6
        classifier.nlp.return_value.ents = [mock_ent]
        result = classifier.clean_resume("OpenAI is a tech company.")
        assert "[ORG]" not in result

    def test_returns_string(self, classifier):
        result = classifier.clean_resume("Experienced Python developer.")
        assert isinstance(result, str)

    def test_whitespace_normalised(self, classifier):
        result = classifier.clean_resume("Python   developer   with   skills.")
        assert "  " not in result


# ── classify_resume ───────────────────────────────────────────────────────────

class TestClassifyResume:
    def test_returns_top_k_categories(self, classifier):
        result = classifier.classify_resume("Python developer with 5 years experience.", top_k=2)
        assert len(result) == 2

    def test_top_1_is_highest_decision_score(self, classifier):
        # decision_function scores are highest for "Technology"
        result = classifier.classify_resume("Python developer with 5 years experience.", top_k=1)
        assert result[0] == "Technology"

    def test_order_is_descending_by_score(self, classifier):
        result = classifier.classify_resume("Some resume.", top_k=3)
        # Based on DECISION_SCORES order: Technology, Engineering, Finance & Accounting
        assert result == ["Technology", "Engineering", "Finance & Accounting"]

    def test_raises_on_top_k_zero(self, classifier):
        with pytest.raises(ValueError, match="top_k must be between"):
            classifier.classify_resume("Some resume.", top_k=0)

    def test_raises_on_top_k_exceeds_classes(self, classifier):
        with pytest.raises(ValueError, match="top_k must be between"):
            classifier.classify_resume("Some resume.", top_k=999)

    def test_top_k_equals_n_classes_allowed(self, classifier):
        n = len(CLASSES)
        # reshape decision_function to return flat array for n classes
        classifier.model.decision_function.return_value = np.array([DECISION_SCORES])
        result = classifier.classify_resume("Some resume.", top_k=n)
        assert len(result) == n

    def test_returns_list_of_strings(self, classifier):
        result = classifier.classify_resume("Some resume.", top_k=2)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_uses_decision_function_not_predict_proba(self, classifier):
        classifier.classify_resume("Some resume.", top_k=1)
        classifier.model.decision_function.assert_called_once()
        classifier.model.predict_proba.assert_not_called()

    def test_clean_resume_called_internally(self, classifier):
        """PII should be stripped before embedding."""
        classifier.classify_resume("Call john.doe@gmail.com for info.", top_k=1)
        call_arg = classifier.embedder.encode.call_args[0][0]
        assert "john.doe@gmail.com" not in call_arg
        assert "[EMAIL]" in call_arg

    def test_embedder_encode_called_once(self, classifier):
        classifier.classify_resume("Some resume.", top_k=1)
        classifier.embedder.encode.assert_called_once()

    def test_all_returned_labels_are_known_classes(self, classifier):
        result = classifier.classify_resume("Some resume.", top_k=3)
        for label in result:
            assert label in CLASSES