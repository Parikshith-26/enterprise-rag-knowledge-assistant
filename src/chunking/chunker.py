import re

from dataclasses import dataclass

from typing import Any

from .cleaner import clean_text


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


def is_heading(line: str) -> bool:
    """
    Detect Markdown headings, numbered headings,
    and common document headings.
    """

    line = line.strip()

    if not line:
        return False

    # Markdown headings
    if re.match(r"^#{1,6}\s+", line):
        return True

    # Numbered headings:
    # 1. Overview
    # 2. Account Access
    # 3.1 Eligibility
    # 4.1.2 Documents Required
    if re.match(
        r"^\d+(?:\.\d+)*\.\s+[A-Z]",
        line
    ):
        return True

    heading_patterns = [
        r"^Key Features$",
        r"^Who Can Apply\??$",
        r"^Documents Required$",
        r"^How to Apply\??$",
        r"^Eligibility$",
        r"^Features$",
        r"^Benefits$",
        r"^Overview$",
        r"^Introduction$",
        r"^Requirements$",
        r"^Customer Support$",
        r"^Contact.*$",
        r"^Frequently Asked Questions$",
        r"^FAQs$",
        r"^Support$",
    ]

    return any(
        re.match(pattern, line, re.IGNORECASE)
        for pattern in heading_patterns
    )


def clean_heading(heading: str) -> str:
    """
    Remove Markdown formatting from headings.
    """

    heading = re.sub(
        r"^#{1,6}\s*\*",
        "",
        heading
    )

    heading = heading.replace("**", "")

    return heading.strip()


def extract_location(
    line: str
) -> tuple[str | None, int | None]:
    """
    Extract page or slide location markers.
    """

    page_match = re.match(
        r"^\[Page\s+(\d+)\]$",
        line,
        re.IGNORECASE
    )

    if page_match:
        return "page", int(page_match.group(1))

    slide_match = re.match(
        r"^\[Slide\s+(\d+)\]$",
        line,
        re.IGNORECASE
    )

    if slide_match:
        return "slide", int(slide_match.group(1))

    return None, None


def split_into_sections(text: str) -> list[dict]:
    """
    Split a document into logical sections based on
    headings and page/slide boundaries.
    """

    lines = text.split("\n")

    sections = []

    current_heading = None
    current_location_type = None
    current_location = None
    current_lines = []

    def save_section():
        if not current_lines:
            return

        content = "\n".join(
            line.strip()
            for line in current_lines
            if line.strip()
        )

        if content:
            sections.append(
                {
                    "heading": current_heading,
                    "text": content,
                    "location_type": current_location_type,
                    "location": current_location,
                }
            )

    for line in lines:

        line = line.strip()

        if not line:
            continue

        location_type, location = extract_location(line)

        if location_type:

            save_section()

            current_lines = []

            # Reset location, but preserve heading behavior
            # across page/slide boundaries.
            current_location_type = location_type
            current_location = location

            continue

        if is_heading(line):

            save_section()

            current_heading = clean_heading(line)

            current_lines = []

            continue

        current_lines.append(line)

    save_section()

    return sections


def split_sentences(text: str) -> list[str]:
    """
    Split text into approximate sentences while
    preserving common abbreviations and decimal values.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9₹])",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def is_question_line(line: str) -> bool:
    """
    Detect FAQ-style questions.
    """

    line = line.strip()

    if not line:
        return False

    # Strongest signal
    if line.endswith("?"):
        return True

    question_patterns = [
        r"^How\s+",
        r"^What\s+",
        r"^Who\s+",
        r"^When\s+",
        r"^Where\s+",
        r"^Why\s+",
        r"^Can\s+",
        r"^Could\s+",
        r"^Is\s+",
        r"^Are\s+",
        r"^Do\s+",
        r"^Does\s+",
    ]

    return any(
        re.match(pattern, line, re.IGNORECASE)
        for pattern in question_patterns
    )


def split_faq_pairs(text: str) -> list[str]:
    """
    Preserve FAQ question-answer pairs.

    Handles cases where PDF extraction places multiple
    FAQ questions and answers in the same paragraph.
    """

    text = text.strip()

    if not text:
        return []

    question_pattern = re.compile(
        r"(?P<question>"
        r"(?:How|What|Who|When|Where|Why|Can|Could|Is|Are|Do|Does)"
        r"\s+[^?]+?\?"
        r")",
        re.IGNORECASE
    )

    matches = list(
        question_pattern.finditer(text)
    )

    if not matches:
        return [text]

    groups = []

    for index, match in enumerate(matches):

        question_start = match.start()
        question_end = match.end()

        if index + 1 < len(matches):

            next_question_start = matches[
                index + 1
            ].start()

            answer = text[
                question_end:next_question_start
            ].strip()

        else:

            answer = text[
                question_end:
            ].strip()

        question = match.group(
            "question"
        ).strip()

        if answer:
            groups.append(
                f"{question} {answer}"
            )
        else:
            groups.append(question)

    return groups


def split_long_text(
    text: str,
    max_words: int,
    overlap_words: int
) -> list[str]:
    """
    Split long text into sentence-aware chunks
    with configurable overlap.
    """

    if len(text.split()) <= max_words:
        return [text]

    sentences = split_sentences(text)

    if not sentences:
        return [text]

    chunks = []

    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence_words = sentence.split()
        sentence_count = len(sentence_words)

        # Handle an individual sentence larger than
        # the maximum chunk size.
        if sentence_count > max_words:

            if current_sentences:

                chunks.append(
                    " ".join(current_sentences)
                )

                current_sentences = []
                current_word_count = 0

            for start in range(
                0,
                sentence_count,
                max_words
            ):

                piece = " ".join(
                    sentence_words[
                        start:start + max_words
                    ]
                )

                chunks.append(piece)

            continue

        # Would adding this sentence exceed
        # the maximum size?
        if (
            current_word_count + sentence_count
            > max_words
            and current_sentences
        ):

            chunks.append(
                " ".join(current_sentences)
            )

            # Build overlap from the end of
            # the previous chunk.
            overlap = []
            overlap_count = 0

            for previous_sentence in reversed(
                current_sentences
            ):

                previous_words = (
                    previous_sentence.split()
                )

                if (
                    overlap_count
                    + len(previous_words)
                    > overlap_words
                ):
                    break

                overlap.insert(
                    0,
                    previous_sentence
                )

                overlap_count += len(
                    previous_words
                )

            current_sentences = overlap
            current_word_count = overlap_count

        current_sentences.append(sentence)

        current_word_count += sentence_count

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


def chunk_document(
    document,
    max_words: int = 350,
    overlap_words: int = 50
) -> list[Chunk]:

    cleaned_text = clean_text(document.text)

    if not cleaned_text:
        return []

    sections = split_into_sections(
        cleaned_text
    )

    # --------------------------------------------------
    # Merge document-level introductory text with the
    # first meaningful section.
    #
    # Prevents a tiny title-only chunk.
    # --------------------------------------------------

    if (
        len(sections) >= 2
        and sections[0]["heading"] is None
        and sections[0]["text"]
    ):

        sections[1]["text"] = (
            sections[0]["text"]
            + "\n"
            + sections[1]["text"]
        )

        sections = sections[1:]

    chunks = []

    chunk_number = 1

    # --------------------------------------------------
    # Document-level searchable context
    # --------------------------------------------------
    #
    # This is the important retrieval improvement.
    #
    # The document title is added to EVERY chunk so
    # semantic search, BM25 and reranking can understand
    # what the chunk belongs to.
    #
    # Example:
    #
    # Document: ATM Locations at Major Petrol Pumps
    # Section: 2. HPCL, Mall Road
    # Mall Road, Kanpur – 208001
    #
    # --------------------------------------------------

    document_title = (
        str(document.title).strip()
        if document.title
        else str(document.document_id).strip()
    )

    document_context = (
        f"Document: {document_title}"
    )

    for section in sections:

        heading = section["heading"]

        section_text = section["text"]

        # --------------------------------------------------
        # FAQ handling
        # --------------------------------------------------

        if heading and (
            "faq" in heading.lower()
            or "frequently asked" in heading.lower()
        ):

            faq_pairs = split_faq_pairs(
                section_text
            )

            section_parts = []

            current_part = []

            current_words = 0

            for faq_pair in faq_pairs:

                pair_words = len(
                    faq_pair.split()
                )

                # Keep question + answer together.
                if (
                    current_part
                    and current_words + pair_words
                    > max_words
                ):

                    section_parts.append(
                        " ".join(current_part)
                    )

                    current_part = []

                    current_words = 0

                current_part.append(
                    faq_pair
                )

                current_words += pair_words

            if current_part:

                section_parts.append(
                    " ".join(current_part)
                )

        else:

            section_parts = split_long_text(
                section_text,
                max_words=max_words,
                overlap_words=overlap_words
            )

        # --------------------------------------------------
        # Create Chunk objects
        # --------------------------------------------------

        for part in section_parts:

            words = part.split()

            chunk_text_parts = []

            # --------------------------------------------------
            # NEW:
            # Always include document title/topic.
            # --------------------------------------------------

            chunk_text_parts.append(
                document_context
            )

            # --------------------------------------------------
            # Preserve section context.
            # --------------------------------------------------

            if heading:

                chunk_text_parts.append(
                    f"Section: {heading}"
                )

            # --------------------------------------------------
            # Actual chunk content.
            # --------------------------------------------------

            chunk_text_parts.append(
                part
            )

            chunk_text = "\n".join(
                chunk_text_parts
            )

            metadata = {
                **document.metadata,
                "source": document.source,
                "title": document.title,
                "document_type": document.document_type,
                "chunk_number": chunk_number,
                "word_count": len(words),

                # Explicit document context
                # is useful for retrieval/debugging.
                "document_context": document_title,
            }

            if heading:

                metadata["section"] = heading

            if section["location_type"]:

                metadata[
                    section["location_type"]
                ] = section["location"]

            chunk_id = (
                f"{document.document_id}"
                f"_chunk_{chunk_number}"
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

            chunk_number += 1

    return chunks