"""ragcheck: measure retrieval before you try to fix it."""

from .chunk import STRATEGIES, Chunk, chunk_document
from .embed import Embedder, HashingEmbedder
from .evaluate import QuestionResult, Report, evaluate
from .ingest import Document, load_corpus
from .store import VectorStore

__all__ = [
    "STRATEGIES",
    "Chunk",
    "Document",
    "Embedder",
    "HashingEmbedder",
    "QuestionResult",
    "Report",
    "VectorStore",
    "chunk_document",
    "evaluate",
    "load_corpus",
]
__version__ = "0.1.0"
