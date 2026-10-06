"""The offsets invariant is the one that matters.

Every chunk must be exactly the slice of the document it claims to be. Once
that holds, a bad retrieval can be traced to a real span of a real file. Once
it stops holding, every debugging session after that is wasted.
"""

from __future__ import annotations

import unittest

from ragcheck.chunk import STRATEGIES, ChunkError, chunk_corpus, chunk_document
from ragcheck.ingest import Document
from tests.support import docs


class TestOffsets(unittest.TestCase):
    def test_every_chunk_is_exactly_its_slice(self):
        for doc in docs():
            for name in STRATEGIES:
                with self.subTest(doc=doc.id, strategy=name):
                    for chunk in chunk_document(doc, name):
                        self.assertEqual(
                            doc.text[chunk.start:chunk.end], chunk.text
                        )

    def test_chunks_are_ordered_and_non_empty(self):
        for doc in docs():
            for name in STRATEGIES:
                with self.subTest(doc=doc.id, strategy=name):
                    chunks = chunk_document(doc, name)
                    self.assertTrue(chunks)
                    starts = [c.start for c in chunks]
                    self.assertEqual(sorted(starts), starts)
                    for chunk in chunks:
                        self.assertTrue(chunk.text.strip())
                        self.assertGreater(chunk.end, chunk.start)

    def test_chunk_ids_are_unique_across_the_corpus(self):
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                ids = [c.id for c in chunk_corpus(docs(), name)]
                self.assertEqual(len(ids), len(set(ids)))

    def test_no_strategy_silently_drops_a_document(self):
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                covered = {c.doc_id for c in chunk_corpus(docs(), name)}
                self.assertEqual({d.id for d in docs()}, covered)


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_chunks(self):
        for doc in docs():
            for name in STRATEGIES:
                with self.subTest(doc=doc.id, strategy=name):
                    first = chunk_document(doc, name)
                    second = chunk_document(doc, name)
                    self.assertEqual(first, second)


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.doc = Document(
            id="d",
            text="# Head\n\nOne two three. Four five six.\n\n## Two\n\nSeven eight.",
        )

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ChunkError, "unknown strategy"):
            chunk_document(self.doc, "semantic")

    def test_fixed_rejects_bad_params(self):
        with self.assertRaises(ChunkError):
            chunk_document(self.doc, "fixed", size=0)
        with self.assertRaises(ChunkError):
            chunk_document(self.doc, "fixed", size=100, overlap=100)

    def test_fixed_overlap_produces_more_chunks(self):
        plain = chunk_document(self.doc, "fixed", size=20, overlap=0)
        lapped = chunk_document(self.doc, "fixed", size=20, overlap=10)
        self.assertGreater(len(lapped), len(plain))

    def test_sentence_never_cuts_mid_sentence(self):
        chunks = chunk_document(self.doc, "sentence", max_size=20)
        for chunk in chunks:
            self.assertFalse(chunk.text.startswith(("two", "three", "five")))

    def test_heading_keeps_the_heading_with_its_body(self):
        chunks = chunk_document(self.doc, "heading")
        self.assertTrue(any(c.text.startswith("# Head") for c in chunks))
        self.assertTrue(any(c.text.startswith("## Two") for c in chunks))

    def test_heading_falls_back_when_there_are_no_headings(self):
        plain = Document(id="p", text="First para.\n\nSecond para.")
        self.assertEqual(2, len(chunk_document(plain, "heading")))

    def test_strategies_disagree_about_the_same_document(self):
        """If they all produced the same chunks there would be nothing to compare."""
        counts = {n: len(chunk_document(self.doc, n)) for n in STRATEGIES}
        self.assertGreater(len(set(counts.values())), 1, counts)


if __name__ == "__main__":
    unittest.main()
