"""Does the harness actually tell good retrieval from bad?

A scoring harness that returns similar numbers whatever you feed it is worse
than useless: it looks like evidence while measuring nothing. These tests
deliberately break retrieval in ways a person would recognise as broken, and
assert the score drops. If any of these stop failing, the harness has stopped
discriminating and nothing else in this repo can be trusted.
"""

from __future__ import annotations

import unittest

from ragcheck.chunk import STRATEGIES, chunk_corpus
from ragcheck.embed import HashingEmbedder
from ragcheck.evaluate import evaluate
from ragcheck.store import VectorStore
from tests.support import docs, questions


def score(strategy, k=5, embedder=None, **params):
    store = VectorStore(embedder or HashingEmbedder())
    store.add(chunk_corpus(docs(), strategy, **params))
    return evaluate(store, questions(), k=k, label=strategy)


class TestDiscrimination(unittest.TestCase):
    def test_absurdly_small_chunks_score_worse_than_sensible_ones(self):
        """40 character windows cut every answer in half. If this does not
        score worse than sentence packing, the harness is not measuring."""
        broken = score("fixed", size=40)
        sensible = score("sentence")
        self.assertLess(broken.hit_rate, sensible.hit_rate)

    def test_one_giant_chunk_per_document_costs_far_more_context(self):
        """Retrieving whole files 'works' and even scores well on precision,
        because relevance is binary per chunk. What it actually does is send
        ten times the text on every call. avg_context_chars is the metric that
        sees it, and it exists because this test failed against precision."""
        giant = score("fixed", size=100000)
        focused = score("sentence")
        self.assertGreater(giant.avg_context_chars,
                           focused.avg_context_chars * 2)

    def test_precision_alone_would_have_missed_that(self):
        """Kept deliberately: it documents why a second cost metric is needed."""
        giant = score("fixed", size=100000)
        focused = score("sentence")
        self.assertGreaterEqual(giant.precision_at_k, focused.precision_at_k)

    def test_strategies_do_not_all_score_the_same(self):
        rates = {name: score(name).hit_rate for name in STRATEGIES}
        self.assertGreater(len(set(rates.values())), 1, rates)

    def test_a_larger_k_never_lowers_hit_rate(self):
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                self.assertLessEqual(score(name, k=3).hit_rate,
                                     score(name, k=10).hit_rate)

    def test_a_tiny_embedding_space_degrades_retrieval(self):
        """16 buckets collides nearly everything. Scores must notice."""
        collided = score("sentence", embedder=HashingEmbedder(dim=16))
        roomy = score("sentence", embedder=HashingEmbedder(dim=1024))
        self.assertLess(collided.mrr, roomy.mrr)

    def test_the_best_strategy_on_this_corpus_is_not_the_obvious_default(self):
        """Paragraph splitting is what most people reach for first. On this
        corpus it is the worst of the four, which is the finding the repo
        exists to make available before you ship rather than after."""
        rates = {name: score(name).hit_rate for name in STRATEGIES}
        self.assertEqual("paragraph", min(rates, key=lambda n: rates[n]), rates)

    def test_scores_are_reproducible_across_runs(self):
        first = score("sentence")
        second = score("sentence")
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_misses_are_named_so_a_bad_average_is_actionable(self):
        report = score("fixed", size=40)
        self.assertTrue(report.misses)
        for result in report.misses:
            self.assertFalse(result.hit)
            self.assertTrue(result.question.id)


if __name__ == "__main__":
    unittest.main()
