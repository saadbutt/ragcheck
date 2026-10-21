"""Embedding and retrieval mechanics.

The default embedder has to be deterministic across machines and processes,
otherwise every score this repo reports is unreproducible and the comparison
it exists to make is worthless.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

from ragcheck.chunk import chunk_corpus
from ragcheck.embed import (
    HashingEmbedder,
    OpenAIEmbedder,
    cosine,
    l2_normalise,
    tokenise,
)
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


class _FakeEmbeddingsServer:
    """A local stand-in for an OpenAI-compatible /embeddings endpoint.

    Records every request body it received and returns a fixed-width vector
    per input, derived from the input's length so different texts provably
    get different vectors without needing a real model.
    """

    def __init__(self, dim: int = 8, fail_with: int | None = None):
        self.dim = dim
        self.fail_with = fail_with
        self.requests: list[dict] = []
        handler = self._make_handler()
        self._httpd = HTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                outer.requests.append({
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": body,
                })
                if outer.fail_with:
                    self.send_response(outer.fail_with)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "boom"}')
                    return
                # Not uniform: a vector of equal components has the same
                # direction regardless of magnitude, so normalising would
                # erase any length signal before a test could see it.
                # Index-reversed on purpose: the embedder must sort by
                # index, not trust response order.
                data = [
                    {"index": i,
                     "embedding": [float(len(text) + 1)] + [1.0] * (outer.dim - 1)}
                    for i, text in enumerate(body["input"])
                ][::-1]
                payload = json.dumps({"data": data}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

        return Handler

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


class TestOpenAIEmbedder(unittest.TestCase):
    """Wire-format and batching tests against a local fake server.

    No real API key or network call, same reasoning as the rest of this
    repo's CI: a test that needs a key is a test that stops running. What's
    actually at risk here isn't the embedding model, it's this file's own
    request shape, its batching arithmetic, and whether it trusts the
    server's ordering -- all of which this can check without one.
    """

    def test_requires_an_api_key(self):
        with self.assertRaises(ValueError):
            OpenAIEmbedder(api_key="")

    def test_request_shape_and_auth_header(self):
        server = _FakeEmbeddingsServer()
        try:
            e = OpenAIEmbedder(api_key="sk-test", model="test-model",
                                base_url=server.base_url, dim=8)
            e.embed(["hello"])
            self.assertEqual(1, len(server.requests))
            req = server.requests[0]
            self.assertEqual("/v1/embeddings", req["path"])
            self.assertEqual("Bearer sk-test", req["auth"])
            self.assertEqual({"model": "test-model", "input": ["hello"]}, req["body"])
        finally:
            server.close()

    def test_response_is_reordered_by_index_not_server_order(self):
        server = _FakeEmbeddingsServer(dim=4)
        try:
            e = OpenAIEmbedder(api_key="sk-test", base_url=server.base_url, dim=4)
            vecs = e.embed(["a", "bb", "ccc"])
            # The fake server deliberately replies index-reversed. If the
            # embedder trusted response order instead of each row's "index",
            # vec[0] (which rises with input length by construction) would
            # come back decreasing instead of increasing.
            self.assertLess(vecs[0][0], vecs[1][0])
            self.assertLess(vecs[1][0], vecs[2][0])
        finally:
            server.close()

    def test_vectors_are_l2_normalised(self):
        server = _FakeEmbeddingsServer(dim=4)
        try:
            e = OpenAIEmbedder(api_key="sk-test", base_url=server.base_url, dim=4)
            for vec in e.embed(["some text"]):
                norm = math.sqrt(sum(v * v for v in vec))
                self.assertAlmostEqual(1.0, norm, places=9)
        finally:
            server.close()

    def test_batches_at_the_configured_size(self):
        server = _FakeEmbeddingsServer(dim=4)
        try:
            e = OpenAIEmbedder(
                api_key="sk-test", base_url=server.base_url, dim=4, batch=2)
            e.embed(["a", "b", "c", "d", "e"])
            # 5 inputs at batch=2 -> 3 requests of sizes 2, 2, 1.
            sizes = [len(r["body"]["input"]) for r in server.requests]
            self.assertEqual([2, 2, 1], sizes)
        finally:
            server.close()

    def test_server_error_propagates_rather_than_being_swallowed(self):
        server = _FakeEmbeddingsServer(fail_with=500)
        try:
            e = OpenAIEmbedder(api_key="sk-test", base_url=server.base_url)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                e.embed(["x"])
            caught.exception.close()
        finally:
            server.close()

    def test_trailing_slash_on_base_url_is_tolerated(self):
        server = _FakeEmbeddingsServer(dim=4)
        try:
            e = OpenAIEmbedder(api_key="sk-test", base_url=server.base_url + "/", dim=4)
            e.embed(["x"])
            self.assertEqual("/v1/embeddings", server.requests[0]["path"])
        finally:
            server.close()


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
