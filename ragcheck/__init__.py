"""ragcheck: measure retrieval before you try to fix it."""

from .chunk import Chunk, STRATEGIES, chunk_document
from .embed import Embedder, HashingEmbedder
from .evaluate import Report, QuestionResult, evaluate
from .ingest import Document, load_corpus
from .store import VectorStore

__all__ = [
    "Chunk", "STRATEGIES", "chunk_document",
    "Embedder", "HashingEmbedder",
    "Report", "QuestionResult", "evaluate",
    "Document", "load_corpus",
    "VectorStore",
]
__version__ = "0.1.0"
