"""Embedding, behind a two-line interface.

The default embedder is deterministic, offline and has no dependencies. That
is a deliberate choice, not a toy: the test suite and CI need to produce the
same retrieval scores on every machine, forever, and a test that needs a
network and an API key is a test that stops running.

HashingEmbedder is a hashed bag of character n-grams. It is genuinely a
retrieval method, roughly lexical, so the numbers the harness reports against
it mean something. It is not as good as a real embedding model on paraphrase,
and the README says so. Swap in OpenAIEmbedder and the same harness reports
the same metrics, which is the entire point of putting an interface here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from typing import List, Protocol, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that turns text into vectors of a fixed width."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


def tokenise(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n == 1:
        return list(tokens)
    return [" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


class HashingEmbedder:
    """Deterministic, offline, dependency free.

    Term frequency over word unigrams and bigrams, hashed into `dim` buckets
    with a stable hash (sha1, not Python's salted hash), then L2 normalised so
    cosine similarity is a dot product.
    """

    def __init__(self, dim: int = 512, ngram_max: int = 2) -> None:
        if dim < 16:
            raise ValueError(f"dim must be at least 16, got {dim}")
        if ngram_max < 1:
            raise ValueError(f"ngram_max must be at least 1, got {ngram_max}")
        self.dim = dim
        self.ngram_max = ngram_max
        self.name = f"hashing-{dim}d-{ngram_max}gram"

    def _bucket(self, term: str) -> int:
        digest = hashlib.sha1(term.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = tokenise(text)
            for n in range(1, self.ngram_max + 1):
                for term in _ngrams(tokens, n):
                    vec[self._bucket(f"{n}:{term}")] += 1.0
            out.append(l2_normalise(vec))
        return out


class OpenAIEmbedder:
    """Any OpenAI-compatible embeddings endpoint. Standard library only.

    Not exercised by the test suite, on purpose. If you point the harness at
    this and your numbers move, that is the harness doing its job.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
        batch: int = 64,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.batch = batch
        self.name = f"openai-{model}"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(self._call(list(texts[i:i + self.batch])))
        return out

    def _call(self, batch: List[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        rows = sorted(body["data"], key=lambda r: r["index"])
        return [l2_normalise(r["embedding"]) for r in rows]


def l2_normalise(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Both sides are normalised on the way in, so this is a dot product.
    Guarded anyway, because an un-normalised vector reaching here should give
    a wrong-looking number rather than a silently plausible one."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))
