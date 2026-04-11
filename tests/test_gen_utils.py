# tests/test_gen_utils.py
import pytest
from lib.gen_utils import clean_text, clean_html


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_basic():
    assert clean_text("Hello World") == "Hello World"

def test_clean_text_strips_disallowed_chars():
    result = clean_text("Hello! World#$%")
    assert "!" not in result
    assert "#" not in result
    assert "$" not in result
    assert "%" not in result

def test_clean_text_preserves_allowed_special_chars():
    text = "C++ dev (senior), 5+ yrs @ Acme - remote/hybrid"
    result = clean_text(text)
    assert "(" in result
    assert ")" in result
    assert "@" in result
    assert "+" in result
    assert "-" in result
    assert "/" in result
    assert "," in result

def test_clean_text_collapses_whitespace():
    result = clean_text("Hello   World\t\there")
    assert "  " not in result
    assert result == "Hello World here"

def test_clean_text_collapses_newlines():
    result = clean_text("line one\nline two\n\nline three")
    assert "\n" not in result
    assert result == "line one line two line three"

def test_clean_text_strips_leading_trailing_whitespace():
    assert clean_text("  hello  ") == "hello"

def test_clean_text_empty_string_raises():
    with pytest.raises(ValueError, match="Text is empty"):
        clean_text("")

def test_clean_text_whitespace_only_raises():
    with pytest.raises(ValueError, match="Text is empty"):
        clean_text("   \t\n  ")

def test_clean_text_preserves_periods_and_dots():
    result = clean_text("B.S. in Computer Science")
    assert "." in result


# ── clean_html ────────────────────────────────────────────────────────────────

def test_clean_html_strips_tags():
    result = clean_html("<p>Hello <b>World</b></p>")
    assert "<" not in result
    assert ">" not in result
    assert "Hello" in result
    assert "World" in result

def test_clean_html_unescapes_entities():
    result = clean_html("&amp; &lt; &gt; &quot;")
    assert "&" in result
    assert "<" in result
    assert ">" in result
    assert '"' in result

def test_clean_html_collapses_whitespace():
    result = clean_html("<p>Hello</p>   <p>World</p>")
    assert "  " not in result

def test_clean_html_empty_string_returns_empty():
    assert clean_html("") == ""

def test_clean_html_no_tags_passthrough():
    result = clean_html("Plain text with no tags")
    assert result == "Plain text with no tags"

def test_clean_html_nested_tags():
    result = clean_html("<div><ul><li>Item one</li><li>Item two</li></ul></div>")
    assert "Item one" in result
    assert "Item two" in result
    assert "<" not in result
