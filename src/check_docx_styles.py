from pathlib import Path

from docx import Document


DOCX_PATH = Path(
    "../RAG-Multi-Corpus-main/"
    "datasets/ZX Bank/docx/(UPI) – ZX Bank Asia.docx"
)


doc = Document(DOCX_PATH)

print("\n========== DOCX STYLES ==========\n")

for paragraph in doc.paragraphs:

    text = paragraph.text.strip()

    if not text:
        continue

    print(
        f"STYLE: {paragraph.style.name}"
    )

    print(
        f"TEXT : {text}"
    )

    print("--------------------------------")