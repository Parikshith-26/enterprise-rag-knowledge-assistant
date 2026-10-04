from pathlib import Path

from .base import Document


def load_markdown(file_path: str, company: str) -> Document:
    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return Document(
        document_id=path.stem,
        source=str(path),
        title=path.stem,
        document_type="markdown",
        text=text.strip(),
        metadata={
            "company": company,
            "file_name": path.name,
        },
    )