# tools/bench

Offline helper for the `ninfer_bench` throughput tool. Correctness/parity tooling lives separately
under [`tools/parity`](../parity).

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v11. The flattened summary and schema-v3 matrix
manifest carry native names from the report: selected target, canonical `weights_id`, artifact,
load/read/upload/staging values, Engine memory arenas including request transient and CUDA Graph
allowance, per-test planned logical and allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

`run_serve_corpus.py` runs both registered targets and both published MTP0/MTP3 suites when both
artifacts are supplied. Pass one `--artifact` to select a single target and `--mode mtp0` or
`--mode mtp3` to run only that suite. The 35B-A3B-only `--mode dflash7` route runs the same
decode corpus with DFlash block=8 (`k=7`) and the optimized proposal head. Add
`--sampling greedy` to force exact argmax while retaining the same fixtures and repetition count.
Its schema-v5 result and flattened summaries retain the canonical `weights_id` received from the
schema-v10 serving startup record. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## KV-cache quality campaign

`run_kv_cache_quality.py` compares BF16, INT8-G64, and KVarN through the real serving route. Its
ledger fixtures use same-schema records, composite-key near misses, conflicting revisions,
multi-record retrieval, and negative controls at fixed depths. Runs are greedy and resumable; raw
responses, exact scores, common-prefix/repetition metrics, request-log timings, and BF16 pairwise
comparisons are retained in the output directory.

```bash
python3 tools/bench/run_kv_cache_quality.py \
  --artifact out/qwen3_8_27b_nvfp4.ninfer \
  --preset standard --mode mtp0 --mode mtp3 \
  --output profiles/bench/kv-cache-quality

python3 tools/bench/run_kv_cache_quality.py \
  --artifact out/qwen3_8_27b_nvfp4.ninfer \
  --loop-only --kv-dtype kvarn --mode mtp3 \
  --max-context 180000 --kv-capacity 360064 --max-concurrency 2 \
  --loop-output-tokens 80000 \
  --output profiles/bench/kv-cache-loop

python3 tools/bench/run_kv_cache_quality.py \
  --artifact out/qwen3_8_27b_nvfp4.ninfer \
  --preset max-needles --needle-count 64 --needle-layout 0 \
  --kv-dtype kvarn --mode mtp3 \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 1 \
  --output profiles/bench/kv-cache-max-needles
```

The default matched comparison uses one explicit logical capacity for every KV format in a mode.
MTP3 defaults to 86,016 tokens because the BF16 cache plus resident MTP state cannot fit the MTP0
98,304-token allocation on a 32 GiB card. `--kv-capacity` is an explicit diagnostic override; use it
only when a target-specific production profile, rather than a matched cross-format comparison, is
the intended claim.

The `max-needles` preset is a KVarN maximum-context qualification unless `--kv-dtype` is supplied
explicitly. It generates three deterministic layouts each with 32 and 64 simultaneous composite-key
needles, distributes them across the complete document with local near-match distractors, and asks
for every value in a permuted order. Before generation, the runner uses the serving token-count
endpoint to choose the largest document that leaves a 2,048-token output budget and 128-token safety
margin under `--max-context`. Reports retain strict whole-response equality, parsed per-needle
recall, misses and incorrect labels, early/middle/late recall, and terminal repetition. Do not claim
a same-context cross-format comparison when another cache format cannot allocate that capacity on
the measured GPU.

## Concurrent serving benchmark

`run_serve_concurrency.py` measures two separate concurrency properties through real loopback
Chat Completions requests:

- `decode-saturation` submits one long-decode wave and uses only complete one-second intervals in
  which every decode round has exactly the configured batch size. Ramp-up, prefill, and drain
  intervals are excluded.
- `corpus-makespan` shuffles the existing mode-specific corpus once with the fixed seed `20260811`,
  then runs that same order with exactly `N` persistent client workers. A worker submits the next
  request only after its current response completes, and makespan ends when the final response has
  been read. Request bodies are sent in shuffled-order sequence while response waits remain fully
  concurrent, removing client-thread arrival races without serializing inference.

Each concurrency point starts a fresh server because its execution graphs and memory plan are
startup-fixed. Prefix reuse is disabled, startup and warmup are outside both measurements, and the
runner writes per-point JSON, raw serving JSONL, and combined JSON/CSV/Markdown summaries.
The point report records the shuffle seed, dispatch method, shuffled position, and canonical corpus
position for every request.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.
