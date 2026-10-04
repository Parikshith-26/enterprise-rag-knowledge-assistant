from pathlib import Path

from .base import Document
from .loader import SUPPORTED_EXTENSIONS, load_document


def load_dataset(
    dataset_path: str,
    company: str
) -> list[Document]:

    root = Path(dataset_path)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {root}"
        )

    documents = []

    for file_path in root.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            document = load_document(
                str(file_path),
                company=company
            )

            documents.append(document)

        except Exception as error:
            print(
                f"Failed to load {file_path}: {error}"
            )

    return documents