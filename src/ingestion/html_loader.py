from pathlib import Path

from bs4 import BeautifulSoup

from .base import Document


def load_html(file_path: str, company: str) -> Document:
    path = Path(file_path)

    html = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that usually don't contain useful knowledge text.
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(
        separator="\n",
        strip=True
    )

    return Document(
        document_id=path.stem,
        source=str(path),
        title=path.stem,
        document_type="html",
        text=text,
        metadata={
            "company": company,
            "file_name": path.name,
        },
    )