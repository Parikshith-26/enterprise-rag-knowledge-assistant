from pathlib import Path

from .base import Document
from .docx_loader import load_docx
from .html_loader import load_html
from .markdown_loader import load_markdown
from .pdf_loader import load_pdf
from .pptx_loader import load_pptx


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".html",
    ".md",
    ".pptx",
}


def load_document(
    file_path: str,
    company: str
) -> Document:

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path, company)

    if extension == ".docx":
        return load_docx(file_path, company)

    if extension == ".html":
        return load_html(file_path, company)

    if extension == ".md":
        return load_markdown(file_path, company)

    if extension == ".pptx":
        return load_pptx(file_path, company)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )