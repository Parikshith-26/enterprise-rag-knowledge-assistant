from pathlib import Path
import re

import pymupdf

from .base import Document


def is_likely_heading(text: str) -> bool:
    """
    Detect common heading patterns in extracted PDF text.
    """

    text = text.strip()

    if not text:
        return False

    # Numbered headings:
    # 1. Online Application
    # 2. Visit a Branch
    # 3. Contact Customer Care
    if re.match(r"^\d+[\.\)]\s+\S+", text):
        return True

    # Numbered sub-headings:
    # 1.1 Something
    # 2.3 Something
    if re.match(r"^\d+\.\d+\s+\S+", text):
        return True

    # Common heading-style phrases
    heading_keywords = [
        "Key Features",
        "Who Can Apply?",
        "Documents Required",
        "Eligibility",
        "How to Apply",
        "Application Process",
        "Benefits",
        "Features",
        "Important Information",
        "Supporting Rural Dreams",
    ]

    for keyword in heading_keywords:
        if text.lower() == keyword.lower():
            return True

    return False


def load_pdf(file_path: str, company: str) -> Document:
    path = Path(file_path)

    pdf = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(pdf, start=1):

        blocks = page.get_text("blocks")

        page_lines = []

        for block in blocks:
            block_text = block[4].strip()

            if not block_text:
                continue

            lines = [
                line.strip()
                for line in block_text.splitlines()
                if line.strip()
            ]

            for line in lines:
                page_lines.append(line)

        if page_lines:

            processed_lines = []

            for line in page_lines:

                if is_likely_heading(line):
                    processed_lines.append(line)
                else:
                    processed_lines.append(line)

            pages.append(
                f"[Page {page_number}]\n"
                + "\n".join(processed_lines)
            )

    pdf.close()

    full_text = "\n\n".join(pages)

    return Document(
        document_id=path.stem,
        source=str(path),
        title=path.stem,
        document_type="pdf",
        text=full_text,
        metadata={
            "company": company,
            "file_name": path.name,
            "page_count": len(pages),
        },
    )