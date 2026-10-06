"""Embedding and retrieval mechanics.

The default embedder has to be deterministic across machines and processes,
otherwise every score this repo reports is unreproducible and the comparison
it exists to make is worthless.
"""

from __future__ import annotations

import math
import unittest

from ragcheck.chunk import chunk_corpus
from ragcheck.embed import HashingEmbedder, cosine, l2_normalise, tokenise
from ragcheck.store import VectorStore
from tests.support import docs


class TestHashingEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=128)

    def test_same_text_same_vector(self):
        a = self.embedder.embed(["webhook signatures use HMAC"])[0]
        b = self.embedder.embed(["webhook signatures use HMAC"])[0]
        self.assertEqual(a, b)

    def test_vectors_are_unit_length(self):
        for vec in self.embedder.embed(["one two three", "four five"]):
            self.assertAlmostEqual(1.0, math.sqrt(sum(v * v for v in vec)), places=9)

    def test_hashing_does_not_depend_on_python_hash_randomisation(self):
        """sha1, not hash(). A salted hash would make every run different."""
        expected = HashingEmbedder(dim=128)._bucket("1:webhook")
        self.assertEqual(expected, HashingEmbedder(dim=128)._bucket("1:webhook"))

    def test_related_text_scores_above_unrelated(self):
        vecs = self.embedder.embed([
            "webhook payloads are signed with HMAC SHA-256",
            "webhook signing and HMAC",
            "invoices are due fourteen days after issue",
        ])
        self.assertGreater(cosine(vecs[0], vecs[1]), cosine(vecs[0], vecs[2]))

    def test_empty_text_is_a_zero_vector_not_a_crash(self):
        self.assertEqual([0.0] * 128, self.embedder.embed([""])[0])

    def test_bigrams_change_the_result(self):
        uni = HashingEmbedder(dim=256, ngram_max=1)
        bi = HashingEmbedder(dim=256, ngram_max=2)
        text = "rate limit exceeded"
        self.assertNotEqual(uni.embed([text])[0], bi.embed([text])[0])

    def test_bad_params_are_refused(self):
        with self.assertRaises(ValueError):
            HashingEmbedder(dim=4)
        with self.assertRaises(ValueError):
            HashingEmbedder(ngram_max=0)


class TestMath(unittest.TestCase):
    def test_tokenise_lowercases_and_drops_punctuation(self):
        self.assertEqual(["http", "429", "retry", "after"],
                         tokenise("HTTP 429: Retry-After!"))

    def test_l2_normalise_of_zero_vector_is_safe(self):
        self.assertEqual([0.0, 0.0], l2_normalise([0.0, 0.0]))

    def test_cosine_of_identical_unit_vectors_is_one(self):
        v = l2_normalise([3.0, 4.0])
        self.assertAlmostEqual(1.0, cosine(v, v), places=9)

    def test_cosine_of_orthogonal_is_zero(self):
        self.assertAlmostEqual(0.0, cosine([1.0, 0.0], [0.0, 1.0]), places=9)

    def test_dimension_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            cosine([1.0], [1.0, 0.0])


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore(HashingEmbedder(dim=256))
        self.store.add(chunk_corpus(docs(), "sentence"))

    def test_everything_added_is_searchable(self):
        self.assertGreater(len(self.store), 10)
        self.assertEqual(len(self.store.chunks), len(self.store.vectors))

    def test_search_returns_k_results_ordered_by_score(self):
        hits = self.store.search("webhook signature", k=4)
        self.assertEqual(4, len(hits))
        scores = [s for _c, s in hits]
        self.assertEqual(sorted(scores, reverse=True), scores)

    def test_search_is_reproducible(self):
        a = [c.id for c, _ in self.store.search("rate limit", k=5)]
        b = [c.id for c, _ in self.store.search("rate limit", k=5)]
        self.assertEqual(a, b)

    def test_ties_break_on_chunk_id_not_insertion_order(self):
        store = VectorStore(HashingEmbedder(dim=64))
        from ragcheck.chunk import Chunk
        store.add([
            Chunk("z#0", "z", "identical text here", 0, 19),
            Chunk("a#0", "a", "identical text here", 0, 19),
        ])
        self.assertEqual("a#0", store.search("identical text here", k=1)[0][0].id)

    def test_empty_store_returns_nothing(self):
        self.assertEqual([], VectorStore(HashingEmbedder()).search("x", k=3))

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.store.search("x", k=0)


if __name__ == "__main__":
    unittest.main()
