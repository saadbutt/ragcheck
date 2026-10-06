"""Metrics, checked against hand-worked numbers.

A scoring harness that is wrong is worse than no harness, because it produces
confident numbers people act on. So these do not test "a float comes back",
they test exact values for retrievals constructed by hand.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from ragcheck.chunk import Chunk
from ragcheck.evaluate import (
    GoldsetError, Question, check_goldset, evaluate, load_goldset,
)
from tests.support import CORPUS, docs, questions


class _FakeStore:
    """Returns a scripted result per question, so ranks are exact."""

    class _E:
        name = "fake"

    def __init__(self, scripted):
        self.scripted = scripted
        self.embedder = self._E()

    def search(self, query, k=5):
        return [(c, 1.0) for c in self.scripted[query][:k]]

    def __len__(self):
        return sum(len(v) for v in self.scripted.values())


def _chunk(doc_id, ordinal, text="body text"):
    return Chunk(f"{doc_id}#{ordinal}", doc_id, text, 0, len(text))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.q1 = Question("q1", "first", ("alpha",))
        self.q2 = Question("q2", "second", ("beta",))

    def test_perfect_retrieval_scores_one(self):
        store = _FakeStore({
            "first": [_chunk("alpha", 0)],
            "second": [_chunk("beta", 0)],
        })
        report = evaluate(store, [self.q1, self.q2], k=1)
        self.assertEqual(1.0, report.hit_rate)
        self.assertEqual(1.0, report.mrr)
        self.assertEqual(1.0, report.precision_at_k)

    def test_total_miss_scores_zero(self):
        store = _FakeStore({
            "first": [_chunk("zeta", 0)],
            "second": [_chunk("zeta", 1)],
        })
        report = evaluate(store, [self.q1, self.q2], k=1)
        self.assertEqual(0.0, report.hit_rate)
        self.assertEqual(0.0, report.mrr)
        self.assertEqual(2, len(report.misses))

    def test_mrr_uses_the_rank_of_the_first_hit(self):
        """q1 hits at rank 3 (1/3), q2 at rank 2 (1/2). Mean is 0.4167."""
        store = _FakeStore({
            "first": [_chunk("zeta", 0), _chunk("zeta", 1), _chunk("alpha", 0)],
            "second": [_chunk("zeta", 2), _chunk("beta", 0), _chunk("zeta", 3)],
        })
        report = evaluate(store, [self.q1, self.q2], k=3)
        self.assertEqual(1.0, report.hit_rate)
        self.assertEqual(round((1 / 3 + 1 / 2) / 2, 4), report.mrr)

    def test_hit_rate_and_mrr_can_disagree(self):
        """Both questions hit, but one is buried. Hit rate hides that, MRR does not."""
        store = _FakeStore({
            "first": [_chunk("alpha", 0)],
            "second": [_chunk("zeta", 0), _chunk("zeta", 1), _chunk("beta", 0)],
        })
        report = evaluate(store, [self.q1, self.q2], k=3)
        self.assertEqual(1.0, report.hit_rate)
        self.assertLess(report.mrr, 1.0)

    def test_precision_counts_every_relevant_chunk_not_just_the_first(self):
        store = _FakeStore({
            "first": [_chunk("alpha", 0), _chunk("alpha", 1), _chunk("zeta", 0)],
        })
        report = evaluate(store, [self.q1], k=3)
        self.assertEqual(round(2 / 3, 4), report.precision_at_k)

    def test_k_truncates_the_retrieval(self):
        store = _FakeStore({
            "first": [_chunk("zeta", 0), _chunk("zeta", 1), _chunk("alpha", 0)],
        })
        self.assertEqual(0.0, evaluate(store, [self.q1], k=2).hit_rate)
        self.assertEqual(1.0, evaluate(store, [self.q1], k=3).hit_rate)


class TestAnswerContains(unittest.TestCase):
    def test_right_document_wrong_span_does_not_count(self):
        """This is the difference between finding the file and finding the answer."""
        question = Question("q", "cost?", ("billing",), ("199 USD per month",))
        wrong = _FakeStore({"cost?": [_chunk("billing", 0, "Invoices are due in 14 days.")]})
        right = _FakeStore({"cost?": [_chunk("billing", 1, "Growth is 199 USD per month.")]})
        self.assertEqual(0.0, evaluate(wrong, [question], k=1).hit_rate)
        self.assertEqual(1.0, evaluate(right, [question], k=1).hit_rate)

    def test_matching_is_case_insensitive(self):
        question = Question("q", "x", ("d",), ("HMAC SHA-256",))
        store = _FakeStore({"x": [_chunk("d", 0, "signed with hmac sha-256 always")]})
        self.assertEqual(1.0, evaluate(store, [question], k=1).hit_rate)

    def test_without_answer_contains_any_chunk_of_the_doc_counts(self):
        question = Question("q", "x", ("d",))
        store = _FakeStore({"x": [_chunk("d", 0, "anything at all")]})
        self.assertEqual(1.0, evaluate(store, [question], k=1).hit_rate)


class TestGoldsetLoading(unittest.TestCase):
    def _write(self, payload):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, tmp)
        tmp.close()
        return pathlib.Path(tmp.name)

    def test_the_example_goldset_loads(self):
        self.assertEqual(16, len(questions()))

    def test_every_goldset_entry_points_at_a_real_document(self):
        check_goldset(questions(), [d.id for d in docs()])

    def test_unknown_document_id_is_caught(self):
        bad = [Question("q", "text", ("not-a-doc",))]
        with self.assertRaisesRegex(GoldsetError, "unknown document id"):
            check_goldset(bad, [d.id for d in docs()])

    def test_not_a_list_is_rejected(self):
        with self.assertRaises(GoldsetError):
            load_goldset(self._write({"q1": "x"}))

    def test_unknown_key_is_rejected_not_ignored(self):
        with self.assertRaisesRegex(GoldsetError, "unknown key"):
            load_goldset(self._write([
                {"question": "x", "relevant_docs": ["a"], "weight": 2}
            ]))

    def test_missing_relevant_docs_is_rejected(self):
        with self.assertRaises(GoldsetError):
            load_goldset(self._write([{"question": "x"}]))

    def test_empty_relevant_docs_is_rejected(self):
        with self.assertRaises(GoldsetError):
            load_goldset(self._write([{"question": "x", "relevant_docs": []}]))

    def test_duplicate_question_id_is_rejected(self):
        with self.assertRaisesRegex(GoldsetError, "duplicate question id"):
            load_goldset(self._write([
                {"id": "q1", "question": "a", "relevant_docs": ["d"]},
                {"id": "q1", "question": "b", "relevant_docs": ["d"]},
            ]))

    def test_empty_goldset_is_rejected(self):
        with self.assertRaises(GoldsetError):
            load_goldset(self._write([]))


if __name__ == "__main__":
    unittest.main()
