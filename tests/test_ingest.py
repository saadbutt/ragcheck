"""Reading documents. Boring on purpose, wrong here poisons everything after."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from ragcheck.ingest import (
    Document,
    IngestError,
    html_to_text,
    load_corpus,
    normalise,
    read_document,
)
from tests.support import CORPUS, docs


class TestNormalise(unittest.TestCase):
    def test_crlf_becomes_lf(self):
        self.assertEqual("a\n\nb", normalise("a\r\n\r\nb"))

    def test_runs_of_blank_lines_collapse(self):
        self.assertEqual("a\n\nb", normalise("a\n\n\n\n\nb"))

    def test_trailing_whitespace_before_newline_goes(self):
        self.assertEqual("a\nb", normalise("a   \nb"))


class TestHtml(unittest.TestCase):
    def test_script_and_style_are_not_indexed(self):
        out = html_to_text(
            "<p>keep this</p><script>var secret=1</script><style>p{color:red}</style>"
        )
        self.assertIn("keep this", out)
        self.assertNotIn("secret", out)
        self.assertNotIn("color", out)

    def test_entities_are_decoded(self):
        self.assertIn("Salt & Stone", html_to_text("<p>Salt &amp; Stone</p>"))


class TestLoadCorpus(unittest.TestCase):
    def test_example_corpus_loads(self):
        loaded = docs()
        self.assertEqual(6, len(loaded))
        self.assertEqual(
            ["api", "billing", "onboarding", "security", "sla", "troubleshooting"],
            sorted(d.id for d in loaded),
        )

    def test_every_document_has_text(self):
        for doc in docs():
            with self.subTest(doc=doc.id):
                self.assertGreater(len(doc.text), 200)

    def test_order_is_stable(self):
        self.assertEqual([d.id for d in docs()], [d.id for d in docs()])

    def test_goldset_json_is_not_ingested_as_a_document(self):
        self.assertNotIn("goldset", [d.id for d in docs()])

    def test_missing_directory_is_an_error(self):
        with self.assertRaises(IngestError):
            load_corpus(CORPUS / "nope")

    def test_empty_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(IngestError, "no supported documents"):
            load_corpus(tmp)

    def test_unsupported_type_is_named_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "data.parquet"
            path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(IngestError, "unsupported file type"):
                read_document(path)

    def test_duplicate_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "a.md").write_text("one", encoding="utf-8")
            (root / "a.txt").write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(IngestError, "duplicate document ids"):
                load_corpus(root)

    def test_empty_id_is_refused(self):
        with self.assertRaises(IngestError):
            Document(id="", text="x")

    def test_missing_pdf_reader_says_so_plainly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "doc.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            try:
                import pypdf  # noqa: F401
                self.skipTest("pypdf is installed, nothing to assert")
            except ImportError:
                with self.assertRaisesRegex(IngestError, "no PDF reader"):
                    read_document(path)


if __name__ == "__main__":
    unittest.main()
