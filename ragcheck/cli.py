"""Command line.

    python -m ragcheck.cli compare corpus
    python -m ragcheck.cli eval corpus --strategy sentence --k 5
    python -m ragcheck.cli query corpus "how are webhooks signed" --k 3

`compare` is the one to run first. It scores every registered chunking
strategy against the same goldset and prints them side by side, which is the
question people actually have and usually answer by guessing.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from .chunk import STRATEGIES, chunk_corpus
from .embed import HashingEmbedder, OpenAIEmbedder
from .evaluate import check_goldset, evaluate, load_goldset
from .ingest import load_corpus
from .store import VectorStore

DEFAULT_GOLDSET = "goldset.json"


def _embedder(args):
    if args.openai_key:
        return OpenAIEmbedder(api_key=args.openai_key, model=args.openai_model)
    return HashingEmbedder(dim=args.dim)


def _load(args):
    corpus = pathlib.Path(args.corpus)
    docs = load_corpus(corpus)
    goldset_path = (
        pathlib.Path(args.goldset) if args.goldset else corpus / DEFAULT_GOLDSET
    )
    questions = load_goldset(goldset_path)
    check_goldset(questions, [d.id for d in docs])
    return docs, questions


def _compare(args) -> int:
    docs, questions = _load(args)
    embedder = _embedder(args)
    print(f"{len(docs)} documents, {len(questions)} questions, "
          f"embedder {embedder.name}, k={args.k}\n")
    header = (f"{'strategy':<14}{'chunks':>8}{'hit rate':>10}"
              f"{'MRR':>8}{'P@k':>8}{'ctx chars':>11}")
    print(header)
    print("-" * len(header))
    rows = []
    for name in STRATEGIES:
        store = VectorStore(embedder)
        store.add(chunk_corpus(docs, name))
        report = evaluate(store, questions, k=args.k, label=name)
        rows.append(report)
        print(f"{name:<14}{report.chunk_count:>8}{report.hit_rate:>10.3f}"
              f"{report.mrr:>8.3f}{report.precision_at_k:>8.3f}"
              f"{report.avg_context_chars:>11.0f}")
    best = max(rows, key=lambda r: (r.hit_rate, r.mrr))
    print(f"\nbest hit rate: {best.label}")
    if best.misses:
        print(f"still missed by {best.label}: "
              f"{', '.join(r.question.id for r in best.misses)}")
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps([r.to_dict() for r in rows], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


def _eval(args) -> int:
    docs, questions = _load(args)
    store = VectorStore(_embedder(args))
    store.add(chunk_corpus(docs, args.strategy))
    report = evaluate(store, questions, k=args.k, label=args.strategy)
    print(f"strategy {args.strategy}, {report.chunk_count} chunks, k={args.k}")
    print(f"hit rate {report.hit_rate:.3f}   MRR {report.mrr:.3f}   "
          f"P@k {report.precision_at_k:.3f}   "
          f"context {report.avg_context_chars:.0f} chars\n")
    for result in report.results:
        rank = result.hit_rank if result.hit else "-"
        mark = "ok " if result.hit else "MISS"
        print(f"{mark} {result.question.id:<5} rank {rank!s:<3} "
              f"{result.question.text[:58]}")
    if report.misses:
        print("\nMissed questions retrieved these instead:")
        for result in report.misses:
            got = ", ".join(c.id for c in result.retrieved[:3]) or "nothing"
            print(f"  {result.question.id}: {got}")
    return 0 if report.hit_rate >= args.min_hit_rate else 1


def _query(args) -> int:
    docs, _ = _load(args) if args.goldset else (load_corpus(args.corpus), None)
    store = VectorStore(_embedder(args))
    store.add(chunk_corpus(docs, args.strategy))
    for rank, (chunk, score) in enumerate(store.search(args.text, k=args.k), 1):
        preview = " ".join(chunk.text.split())[:110]
        print(f"{rank}. {chunk.id}  score {score:.3f}  [{chunk.start}:{chunk.end}]")
        print(f"   {preview}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragcheck")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("corpus", type=str)
        p.add_argument("--goldset", type=str, default="")
        p.add_argument("--k", type=int, default=5)
        p.add_argument("--dim", type=int, default=512)
        p.add_argument("--openai-key", type=str, default="")
        p.add_argument("--openai-model", type=str,
                       default="text-embedding-3-small")

    c = sub.add_parser("compare", help="score every chunking strategy")
    common(c)
    c.add_argument("--json", type=str, default="")

    e = sub.add_parser("eval", help="score one strategy, per question")
    common(e)
    e.add_argument("--strategy", choices=sorted(STRATEGIES), default="paragraph")
    e.add_argument("--min-hit-rate", type=float, default=0.0,
                   help="exit non-zero below this, for CI")

    q = sub.add_parser("query", help="retrieve for one question")
    common(q)
    q.add_argument("text", type=str)
    q.add_argument("--strategy", choices=sorted(STRATEGIES), default="paragraph")

    args = parser.parse_args(argv)
    if args.cmd == "compare":
        return _compare(args)
    if args.cmd == "eval":
        return _eval(args)
    return _query(args)


if __name__ == "__main__":
    raise SystemExit(main())
