import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving
    paragraph boundaries.
    """

    if not text:
        return ""

    # Normalize Windows line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces at the beginning/end of each line.
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # Remove blank lines at the beginning/end.
    return text.strip()