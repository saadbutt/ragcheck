"""End to end through the command a reviewer would actually run."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest

from ragcheck import cli
from tests.support import ROOT


class TestCli(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        os.chdir(ROOT)

    def tearDown(self):
        os.chdir(self._cwd)

    def test_compare_runs_every_strategy(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["compare", "corpus"])
        out = buf.getvalue()
        self.assertEqual(0, code)
        for name in ("fixed", "paragraph", "sentence", "heading"):
            self.assertIn(name, out)
        self.assertIn("best hit rate", out)

    def test_compare_writes_json_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "out.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["compare", "corpus", "--json", str(target)])
            rows = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(4, len(rows))
            for row in rows:
                self.assertIn("hit_rate", row)
                self.assertIn("misses", row)

    def test_eval_lists_every_question(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["eval", "corpus", "--strategy", "sentence"])
        out = buf.getvalue()
        self.assertIn("q1", out)
        self.assertIn("q16", out)

    def test_eval_exit_code_gates_on_min_hit_rate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            passing = cli.main(["eval", "corpus", "--strategy", "sentence",
                                "--min-hit-rate", "0.5"])
            failing = cli.main(["eval", "corpus", "--strategy", "sentence",
                                "--min-hit-rate", "0.99"])
        self.assertEqual(0, passing)
        self.assertEqual(1, failing)

    def test_query_prints_offsets_so_a_hit_can_be_traced(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["query", "corpus", "how are webhooks signed",
                      "--strategy", "sentence", "--k", "2"])
        out = buf.getvalue()
        self.assertIn("api#", out)
        self.assertRegex(out, r"\[\d+:\d+\]")


if __name__ == "__main__":
    unittest.main()
