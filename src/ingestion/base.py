from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    document_id: str
    source: str
    title: str
    document_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)