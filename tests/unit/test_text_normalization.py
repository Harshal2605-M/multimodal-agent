from app.utils.text_normalization import normalize_text


def test_normalize_text_removes_null_bytes() -> None:
    assert normalize_text(
        "hello\x00 world"
    ) == "hello world"


def test_normalize_text_normalizes_line_endings() -> None:
    assert normalize_text(
        "first\r\nsecond\rthird"
    ) == "first\nsecond\nthird"


def test_normalize_text_collapses_horizontal_whitespace() -> None:
    assert normalize_text(
        "Revenue     increased\tby 20%."
    ) == "Revenue increased by 20%."


def test_normalize_text_preserves_paragraph_boundary() -> None:
    text = (
        "First paragraph.\n\n\n\n"
        "Second paragraph."
    )

    assert normalize_text(text) == (
        "First paragraph.\n\n"
        "Second paragraph."
    )


def test_normalize_text_strips_line_whitespace() -> None:
    text = "  first line  \n  second line  "

    assert normalize_text(text) == (
        "first line\nsecond line"
    )


def test_normalize_empty_text() -> None:
    assert normalize_text("") == ""