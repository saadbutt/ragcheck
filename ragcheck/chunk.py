"""Chunking strategies.

This is the knob that decides whether retrieval works, and it is the knob
people turn blind. Every strategy here is a pure function of (text, params),
every chunk carries the offsets it came from, and `document.text[start:end]`
is always exactly `chunk.text`. That invariant is what lets a bad retrieval
be traced back to a real span of a real document instead of a guess.

Add a strategy by writing a function and registering it in STRATEGIES. The
evaluation harness picks strategies up from that dict, so a new one is
comparable against the others without touching evaluate.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List

from .ingest import Document


class ChunkError(ValueError):
    """Raised when a strategy is asked for something impossible."""


@dataclass(frozen=True)
class Chunk:
    """A span of one document.

    id is `<doc_id>#<ordinal>`, which is what a goldset refers to when it
    wants to be precise about where an answer lives.
    """

    id: str
    doc_id: str
    text: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARA_SPLIT = re.compile(r"\n\s*\n")
_MD_HEADING = re.compile(r"^#{1,6}\s+.+$", re.M)


def _emit(doc: Document, spans: List[tuple]) -> List[Chunk]:
    """Turn (start, end) pairs into Chunks, dropping empties.

    Offsets are preserved exactly. Nothing here re-slices or re-joins text,
    because that is how offsets quietly stop matching the source.
    """
    chunks: List[Chunk] = []
    for start, end in spans:
        text = doc.text[start:end]
        if not text.strip():
            continue
        lead = len(text) - len(text.lstrip())
        trail = len(text) - len(text.rstrip())
        start += lead
        end -= trail
        chunks.append(
            Chunk(
                id=f"{doc.id}#{len(chunks)}",
                doc_id=doc.id,
                text=doc.text[start:end],
                start=start,
                end=end,
            )
        )
    return chunks


def fixed(doc: Document, size: int = 600, overlap: int = 0) -> List[Chunk]:
    """Fixed character windows. The default everyone starts with, and the
    reason retrieval so often returns a sentence cut in half."""
    if size <= 0:
        raise ChunkError(f"size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ChunkError(f"overlap must be in [0, size), got {overlap}")
    step = size - overlap
    spans = [(i, min(i + size, len(doc.text)))
             for i in range(0, max(len(doc.text), 1), step)]
    return _emit(doc, spans)


def paragraph(doc: Document, max_size: int = 1200) -> List[Chunk]:
    """Split on blank lines. Paragraphs longer than max_size are split again
    on sentences, so one runaway paragraph cannot swallow the document."""
    spans: List[tuple] = []
    cursor = 0
    for part in _PARA_SPLIT.split(doc.text):
        start = doc.text.index(part, cursor) if part else cursor
        end = start + len(part)
        cursor = end
        if len(part) <= max_size:
            spans.append((start, end))
            continue
        spans.extend(_sentence_spans(doc.text, start, end, max_size))
    return _emit(doc, spans)


def sentence(doc: Document, max_size: int = 400) -> List[Chunk]:
    """Pack whole sentences up to max_size. Never cuts mid-sentence."""
    return _emit(doc, _sentence_spans(doc.text, 0, len(doc.text), max_size))


def _sentence_spans(text: str, start: int, end: int, max_size: int) -> List[tuple]:
    body = text[start:end]
    spans: List[tuple] = []
    cursor = start
    buf_start = start
    for piece in _SENTENCE_END.split(body):
        if not piece:
            continue
        piece_start = text.index(piece, cursor)
        piece_end = piece_start + len(piece)
        cursor = piece_end
        if piece_end - buf_start > max_size and piece_start > buf_start:
            spans.append((buf_start, piece_start))
            buf_start = piece_start
    if cursor > buf_start:
        spans.append((buf_start, cursor))
    return spans


def heading(doc: Document, max_size: int = 2000) -> List[Chunk]:
    """Markdown heading sections. A section keeps its heading with its body,
    which is usually what makes a chunk answerable on its own."""
    positions = [m.start() for m in _MD_HEADING.finditer(doc.text)]
    if not positions:
        return paragraph(doc, max_size=max_size)
    if positions[0] != 0:
        positions.insert(0, 0)
    bounds = positions + [len(doc.text)]
    spans: List[tuple] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end - start <= max_size:
            spans.append((start, end))
        else:
            spans.extend(_sentence_spans(doc.text, start, end, max_size))
    return _emit(doc, spans)


STRATEGIES: Dict[str, Callable[..., List[Chunk]]] = {
    "fixed": fixed,
    "paragraph": paragraph,
    "sentence": sentence,
    "heading": heading,
}


def chunk_document(doc: Document, strategy: str = "paragraph", **params) -> List[Chunk]:
    try:
        fn = STRATEGIES[strategy]
    except KeyError:
        raise ChunkError(
            f"unknown strategy {strategy!r}, have {sorted(STRATEGIES)}"
        ) from None
    return fn(doc, **params)


def chunk_corpus(docs: List[Document], strategy: str = "paragraph", **params):
    out: List[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, strategy, **params))
    return out
