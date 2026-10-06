"""The point of this repo.

You cannot fix retrieval you cannot measure, and "it seems better" is not a
measurement. This module turns a goldset and a retriever into numbers:
hit rate, recall, precision and MRR, with a per-question breakdown so a bad
average can be traced to the questions that caused it.

One design decision matters more than the rest. A goldset entry names the
DOCUMENT the answer lives in, never a chunk id. Chunk ids change the moment
you change chunking, and a goldset keyed on them can only ever score the
chunking it was written against, which makes comparison impossible. Naming
documents keeps one goldset valid across every strategy.

`answer_contains` narrows it further without re-introducing that problem: a
retrieved chunk counts only if it comes from the right document AND actually
contains the answer text. That is what separates "retrieved the right file"
from "retrieved the part of it that answers the question", and it is usually
where a pipeline is quietly failing.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .chunk import Chunk


class GoldsetError(ValueError):
    """Raised when a goldset is malformed. Never guessed around."""


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    relevant_docs: Tuple[str, ...]
    answer_contains: Tuple[str, ...] = ()

    def matches(self, chunk: Chunk) -> bool:
        if chunk.doc_id not in self.relevant_docs:
            return False
        if not self.answer_contains:
            return True
        body = chunk.text.lower()
        return any(phrase.lower() in body for phrase in self.answer_contains)


@dataclass
class QuestionResult:
    question: Question
    retrieved: List[Chunk]
    hit_rank: Optional[int]          # 1-based rank of the first match, else None

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.hit_rank is None else 1.0 / self.hit_rank

    @property
    def matched(self) -> int:
        return sum(1 for c in self.retrieved if self.question.matches(c))


@dataclass
class Report:
    """Aggregate plus the per-question detail that makes it actionable."""

    label: str
    k: int
    results: List[QuestionResult] = field(default_factory=list)
    chunk_count: int = 0
    embedder: str = ""

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def hit_rate(self) -> float:
        """Share of questions with at least one correct chunk in the top k.
        The number to quote to a client, because it answers "how often does
        the model get anything useful to work with"."""
        return self._mean(1.0 if r.hit else 0.0 for r in self.results)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank. Falls when correct chunks are retrieved but
        buried, which hit rate alone hides."""
        return self._mean(r.reciprocal_rank for r in self.results)

    @property
    def precision_at_k(self) -> float:
        """Share of retrieved chunks that were relevant. Low precision means
        you are filling the context window with noise and paying for it."""
        return self._mean(
            (r.matched / len(r.retrieved)) if r.retrieved else 0.0
            for r in self.results
        )

    @property
    def avg_context_chars(self) -> float:
        """Characters handed to the model per question.

        This exists because precision@k cannot see it. Relevance is binary per
        chunk, so a strategy that returns one whole document per chunk scores
        well on precision while quietly sending ten times the text. That is
        paid for in tokens on every single call. A failing test is what
        surfaced this, and the metric was added rather than the test softened.
        """
        return self._mean(
            float(sum(len(c.text) for c in r.retrieved)) for r in self.results
        )

    @property
    def misses(self) -> List[QuestionResult]:
        return [r for r in self.results if not r.hit]

    def _mean(self, values) -> float:
        vals = list(values)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "k": self.k,
            "embedder": self.embedder,
            "questions": self.total,
            "chunks": self.chunk_count,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "precision_at_k": self.precision_at_k,
            "avg_context_chars": self.avg_context_chars,
            "misses": [r.question.id for r in self.misses],
        }


def load_goldset(path) -> List[Question]:
    """Read goldset.json. Every field is checked; a typo in a document id is
    the single easiest way to get a confidently wrong score."""
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise GoldsetError("goldset must be a JSON list of question objects")
    questions: List[Question] = []
    seen = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise GoldsetError(f"question {i} must be an object")
        extra = sorted(set(item) - {"id", "question", "relevant_docs",
                                    "answer_contains"})
        if extra:
            raise GoldsetError(f"question {i}: unknown key(s) {extra}")
        qid = str(item.get("id") or f"q{i + 1}")
        if qid in seen:
            raise GoldsetError(f"duplicate question id: {qid}")
        seen.add(qid)
        text = item.get("question")
        if not isinstance(text, str) or not text.strip():
            raise GoldsetError(f"{qid}: question must be a non-empty string")
        docs = item.get("relevant_docs")
        if not isinstance(docs, list) or not docs:
            raise GoldsetError(f"{qid}: relevant_docs must be a non-empty list")
        if not all(isinstance(d, str) and d for d in docs):
            raise GoldsetError(f"{qid}: relevant_docs must be strings")
        contains = item.get("answer_contains", [])
        if not isinstance(contains, list) or not all(
            isinstance(c, str) for c in contains
        ):
            raise GoldsetError(f"{qid}: answer_contains must be a list of strings")
        questions.append(
            Question(qid, text.strip(), tuple(docs), tuple(contains))
        )
    if not questions:
        raise GoldsetError("goldset is empty")
    return questions


def check_goldset(questions: Sequence[Question], doc_ids: Sequence[str]) -> None:
    """Fail loudly when a goldset names a document the corpus does not have."""
    known = set(doc_ids)
    unknown = sorted({
        d for q in questions for d in q.relevant_docs if d not in known
    })
    if unknown:
        raise GoldsetError(
            f"goldset references unknown document id(s): {unknown}. "
            f"corpus has: {sorted(known)}"
        )


def evaluate(store, questions: Sequence[Question], k: int = 5,
             label: str = "") -> Report:
    """Run every question through the store and score the results."""
    report = Report(
        label=label or getattr(store.embedder, "name", "store"),
        k=k,
        chunk_count=len(store),
        embedder=getattr(store.embedder, "name", ""),
    )
    for question in questions:
        retrieved = [chunk for chunk, _score in store.search(question.text, k=k)]
        hit_rank = None
        for rank, chunk in enumerate(retrieved, start=1):
            if question.matches(chunk):
                hit_rank = rank
                break
        report.results.append(QuestionResult(question, retrieved, hit_rank))
    return report
