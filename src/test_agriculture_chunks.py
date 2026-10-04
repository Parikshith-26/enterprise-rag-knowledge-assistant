import json
from pathlib import Path


METADATA_PATH = Path(
    "data/indexes/zx_bank_metadata.json"
)


def main():

    print("\n========== AGRICULTURE LOAN CHUNK INSPECTION ==========\n")

    with METADATA_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    matches = []

    for record in metadata:

        text = record.get(
            "text",
            ""
        )

        title = record.get(
            "title",
            ""
        )

        document_id = record.get(
            "document_id",
            ""
        )

        section = record.get(
            "metadata",
            {}
        ).get(
            "section",
            ""
        )

        searchable_text = " ".join(
            [
                str(title),
                str(document_id),
                str(section),
                text,
            ]
        ).lower()

        if (
            "agriculture" in searchable_text
            or "agri-loan" in searchable_text
        ):
            matches.append(record)

    print(
        f"Agriculture-related chunks found: "
        f"{len(matches)}"
    )

    for number, record in enumerate(
        matches,
        start=1
    ):

        metadata = record.get(
            "metadata",
            {}
        )

        print("\n" + "=" * 80)

        print(
            f"RESULT {number}"
        )

        print(
            f"Document: "
            f"{record.get('title', 'N/A')}"
        )

        print(
            f"Document ID: "
            f"{record.get('document_id', 'N/A')}"
        )

        print(
            f"Chunk ID: "
            f"{record.get('chunk_id', 'N/A')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section', 'N/A')}"
        )

        print("\nTEXT:")

        print(
            record.get(
                "text",
                ""
            )
        )

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()