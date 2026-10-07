# ragcheck

Measure retrieval before you try to fix it.

Most RAG work stops at "it returns something". The question nobody answers is
whether the right passage was retrieved, how often, how far down the list, and
what it cost in context. ragcheck answers those four with numbers, and lets you
compare chunking strategies against the same goldset instead of guessing.

```
$ python -m ragcheck.cli compare corpus
6 documents, 16 questions, embedder hashing-512d-2gram, k=5

strategy        chunks  hit rate     MRR     P@k  ctx chars
-----------------------------------------------------------
fixed               12     0.812   0.688   0.163       2490
paragraph           62     0.500   0.323   0.100        691
sentence            17     0.875   0.613   0.175       1736
heading             33     0.812   0.612   0.163        925

best hit rate: sentence
still missed by sentence: q4, q15
```

Paragraph splitting is what most people reach for first. On this corpus it is
the worst of the four by a wide margin, and you would never find that out by
reading the output and nodding.

Python 3.9 or newer. No required dependencies, including for the tests.

## Run it

```
python -m ragcheck.cli compare corpus
python -m ragcheck.cli eval corpus --strategy sentence
python -m ragcheck.cli query corpus "how are webhooks signed" --k 3
python -m unittest discover -s tests -t .
```

`eval` prints every question with the rank its first correct chunk landed at,
then lists what the misses retrieved instead. That list is the actual work: an
average tells you something is wrong, the miss list tells you what.

```
ok  q1    rank 1   How much does the Growth plan cost per month?
ok  q2    rank 5   What happens to my allowance if I downgrade my plan?
MISS q4    rank -   What is the rate limit for a Growth account?
```

## The four numbers

**hit rate** is the share of questions where at least one correct chunk made
the top k. This is the one to quote, because it answers "how often does the
model get anything useful to work with".

**MRR** is where the first correct chunk ranked. Hit rate can sit at 1.0 while
every answer is buried at position five, and MRR is what shows that.

**precision@k** is the share of retrieved chunks that were relevant.

**ctx chars** is how much text got handed to the model per question. It exists
because precision could not see it, which is covered below.

## The goldset names documents, never chunks

This is the one design decision the repo turns on.

Chunk ids change the moment you change chunking. A goldset keyed on them can
only ever score the chunking it was written against, which makes comparison
impossible, which removes the entire reason to have a harness.

So a goldset entry names the **document** the answer lives in:

```json
{
  "id": "q4",
  "question": "What is the rate limit for a Growth account?",
  "relevant_docs": ["api"],
  "answer_contains": ["60 requests per second"]
}
```

`answer_contains` narrows it without reintroducing the problem. A retrieved
chunk counts only if it comes from the right document **and** actually contains
the answer. That is the difference between finding the right file and finding
the part of it that answers the question, and it is usually where a pipeline is
quietly failing.

## What writing the tests changed

The first version scored whole-document chunks as good retrieval. Precision@k
rated them highly, because relevance is binary per chunk: a chunk containing
the answer is relevant whether it is 200 characters or 20,000.

A test asserting that giant chunks should score worse failed, correctly. The
metric was wrong, not the test. Returning entire documents does work, and it
also sends 2.7 times the text on every single call, which you pay for in tokens
forever.

So `avg_context_chars` was added rather than the test softened, and the test
that would have missed it is still in the suite, named as such, to document
why a second cost metric is needed.

## The default embedder is offline on purpose

`HashingEmbedder` is term frequency over word unigrams and bigrams, hashed into
fixed buckets with sha1 and L2 normalised. No network, no key, no dependency,
and identical output on every machine forever.

That matters because a test that needs an API key is a test that stops running,
and a harness whose numbers are not reproducible cannot be used to argue that a
change helped.

It is roughly lexical, so it is weaker than a real embedding model on
paraphrase. `OpenAIEmbedder` is there for real runs, standard library only, and
the same harness reports the same four metrics over it. If your numbers move
when you swap, that is the harness doing its job.

## Adding a chunking strategy

Write a function, register it in `STRATEGIES`:

```python
def semantic(doc, threshold=0.8):
    ...
    return chunks

STRATEGIES["semantic"] = semantic
```

`compare` picks it up from that dict. Nothing in `evaluate.py` changes, and the
new strategy is scored against the same goldset as the existing four.

Every chunk carries the offsets it came from, and
`document.text[chunk.start:chunk.end] == chunk.text` is asserted for every
strategy against every document in the test suite. Once that holds, a bad
retrieval traces back to a real span of a real file. Once it stops holding,
every debugging session after it is wasted.

## Tests

77 tests, no dependencies, under a second.

| file | what it holds up |
| --- | --- |
| `test_ingest.py` | reading, normalising, and refusing what it cannot read |
| `test_chunk.py` | the offsets invariant, determinism, per-strategy behaviour |
| `test_embed_store.py` | reproducible vectors, cosine correctness, ties break on id |
| `test_evaluate.py` | metrics against hand-worked numbers, goldset validation |
| `test_discriminates.py` | the harness actually separates good retrieval from bad |
| `test_cli.py` | end to end through the command a reviewer would run |

`test_discriminates.py` is the one worth reading. A scoring harness that
returns similar numbers whatever you feed it is worse than no harness, because
it looks like evidence while measuring nothing. Those tests break retrieval in
ways a person would recognise as broken (40 character windows, a 16 bucket
embedding space, one chunk per document) and assert the score notices.

## Using it on your own corpus

Put your documents in a folder, write a `goldset.json` beside them, run
`compare`. Twenty questions is enough to be useful and takes about an hour to
write honestly. Writing them yourself is the point: the questions are where
your knowledge of the domain enters the measurement.

`eval --min-hit-rate 0.8` exits non-zero below the threshold, so retrieval
quality can gate a build the same way tests do.

## License

MIT.
# ragcheck
