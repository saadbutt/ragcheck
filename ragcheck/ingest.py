"""Read a corpus off disk into Documents.

Deliberately boring. The interesting part of this repo is evaluate.py, and
ingestion exists so that the evaluation has something real to run against.

Text, Markdown and HTML are handled with the standard library. PDF is handled
only if a reader is installed, and says so plainly rather than failing late.
"""

from __future__ import annotations

import html.parser
import pathlib
import re
from dataclasses import dataclass, field

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
HTML_SUFFIXES = {".html", ".htm"}


class IngestError(ValueError):
    """Raised when a document cannot be read."""


@dataclass(frozen=True)
class Document:
    """One source document. `id` is the stable handle used by the goldset."""

    id: str
    text: str
    path: str = ""
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise IngestError("document id must not be empty")


class _Strip(html.parser.HTMLParser):
    """Visible text only. Script and style content is dropped, not indexed."""

    SKIP = frozenset({"script", "style", "noscript", "head"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth:
            return
        chunk = data.strip()
        if chunk:
            self.parts.append(chunk)


def html_to_text(source: str) -> str:
    parser = _Strip()
    parser.feed(source)
    return "\n\n".join(parser.parts)


def normalise(text: str) -> str:
    """Collapse carriage returns and runs of blank lines. Offsets are taken
    against the normalised text, so this runs once, at read time."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_document(path: pathlib.Path, doc_id: str = "") -> Document:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        raw = path.read_text(encoding="utf-8")
    elif suffix in HTML_SUFFIXES:
        raw = html_to_text(path.read_text(encoding="utf-8"))
    elif suffix == ".pdf":
        raw = _read_pdf(path)
    else:
        raise IngestError(f"unsupported file type: {path.name}")
    return Document(id=doc_id or path.stem, text=normalise(raw), path=str(path))


def _read_pdf(path: pathlib.Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        raise IngestError(
            f"{path.name} is a PDF and no PDF reader is installed. "
            "Install pypdf, or convert the file first. "
            "ragcheck has no required dependencies on purpose."
        ) from None
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_corpus(directory) -> list[Document]:
    """Every supported file in `directory`, sorted by name so runs repeat."""
    root = pathlib.Path(directory)
    if not root.is_dir():
        raise IngestError(f"not a directory: {root}")
    supported = TEXT_SUFFIXES | HTML_SUFFIXES | {".pdf"}
    docs = [
        read_document(p)
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in supported
    ]
    if not docs:
        raise IngestError(f"no supported documents in {root}")
    ids = [d.id for d in docs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise IngestError(f"duplicate document ids: {duplicates}")
    return docs
