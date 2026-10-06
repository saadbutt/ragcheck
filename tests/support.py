"""Shared fixtures. Tests run against the real corpus in corpus/, not toys."""

from __future__ import annotations

import pathlib

from ragcheck.evaluate import load_goldset
from ragcheck.ingest import load_corpus

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"


def docs():
    return load_corpus(CORPUS)


def questions():
    return load_goldset(CORPUS / "goldset.json")
