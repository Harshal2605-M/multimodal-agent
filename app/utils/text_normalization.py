import re


NULL_BYTE = "\x00"

HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[^\S\r\n]+")

EXCESSIVE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Normalize extracted or user-provided text without destroying
    meaningful paragraph boundaries.
    """

    if not text:
        return ""

    normalized = text.replace(NULL_BYTE, "")

    normalized = normalized.replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")

    normalized = HORIZONTAL_WHITESPACE_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = "\n".join(
        line.strip()
        for line in normalized.splitlines()
    )

    normalized = EXCESSIVE_BLANK_LINES_PATTERN.sub(
        "\n\n",
        normalized,
    )

    return normalized.strip()