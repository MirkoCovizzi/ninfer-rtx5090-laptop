---
name: kvarn-production
description: Use for KVarN implementation, optimization, qualification, or review in NInfer.
---

# KVarN Production Workflow

## Fixed Contract

- Target Huawei KVarN K4V2-G64 only: D256, group 64, K4/V2, 8 Sinkhorn passes.
- Treat Huawei commit `7586257f1c632e63187bfacbbe21ccb51540f7b3` as the codec/layout reference.
- Keep KVarN implementation under `src/ops/kvarn/`. Do not add KVarN branches, policies, or loaders
  to BF16/INT8 kernels. Share only identity-free primitives and reducers.
- Preserve official record bits, NInfer page translation, sink/tail lifecycle, and public Engine
  semantics. Do not add generic bit-width machinery.
- Work directly without subagents. Make one focused change at a time.

## Required Order

1. State one hypothesis, one affected phase (`codec`, `lifecycle`, `decode`, or `prefill`), and the
   observable success criterion. Record the current commit and baseline numbers. Commit only when
   requested.
2. Inspect only the owning KVarN code, applicable Op contract, independent oracle, and official
   source. Do not refactor neighboring paths.
3. Implement the smallest coherent KVarN-owned change. Remove failed/superseded experimental code;
   do not retain alternate production routes.
4. Build before benchmarking:

   ```bash
   cmake --build build --target ninfer_kvarn_test ninfer_kvarn_attention_bench \
      ninfer_softmax_attention_test -j
   ```

5. Pass numerical and regression gates:

   ```bash
   ./build/tests/ninfer_kvarn_test
    ./build/tests/ninfer_softmax_attention_test
   git diff --check
   ```

   The KVarN oracle must cover represented stored bits, sink/compressed/tail reads, H24/KV4 and
   H16/KV2, width 1 and width 6, and the affected tiled route. If state ownership changes, also run
    `ninfer_qwen3_6_runtime_mechanisms_test`. If capture behavior changes, require CUDA Graph replay.
    Staging checks must compare exact BF16 tails against independently transformed inputs, not
    against a cache already produced by the same staging kernel. Cover page closure, rejection,
    and re-encoding with both scalar and packed verification. Reject non-finite outputs explicitly.
6. Run matched warm/eager H24/KV4 microbenchmarks at 8K, 32K, 128K, and 192K. Decode is W=1;
   prefill is a 1,024-token append-and-attend chunk ending at the stated context. Compare median,
   min, and p95 against both the recorded KVarN baseline and identical INT8 work. Do not infer
    end-to-end performance from operator latency.
    The attention benchmark restores inputs and pre-append state outside timing. Use both closing
    and non-closing contexts (for example 8192 and 8198); compare the printed `phase`, `width`, and
    `closes_page`. Full Engine gates must span several page boundaries, with at least 128 decode
    tokens and a non-page-aligned prompt. Record acceptance alongside throughput.
7. Keep the change only if the targeted metric improves without a material regression elsewhere.
   If it regresses, profile the regression before revising; do not stack another speculative change.
8. Use Nsight Compute only after a measured win or to explain a measured regression. At 192K collect:

   - `SpeedOfLight`, `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`
   - `LaunchStats`, `Occupancy`, `SchedulerStats`, `WarpStateStats`
   - `InstructionStats`, `SourceCounters`

   Save `.ncu-rep` under `profiles/bench/`. Compare KVarN and INT8 kernel geometry, registers/thread,
   shared memory, achieved occupancy, eligible/no-eligible cycles, scoreboard stalls, instructions,
   excessive global/shared sectors, tensor utilization, and DRAM utilization. Attribute the win to
   changed counters, not timing alone.
9. Rebuild `ninfer-serve` only after operator qualification. Smoke the real artifact, CUDA Graph,
   MTP width 6, prefix restore, and concurrent rows affected by the change.
   For cache-transition changes, build `ninfer_qwen3_6_27b_mtp_greedy_parity_real_test` and run it
   with `--kvarn-only` and explicit `NINFER_MTP_GREEDY_PARITY_WEIGHTS`. This checks exact generated
   token parity for MTP0 through MTP5 over 8,192 generated tokens by default, one run per case/row/depth.
   Use the targeted matrix in `docs/maintainer/paged-kv-cache.md` for mixed concurrent rows,
   confirmed prefix restoration, eager/full-head execution, and long resident contexts. Do not
   substitute a 128-token smoke for the long-decode gate. Packed groups must preserve each query's
   scalar split partition as well as its page representation.
10. Run quality qualification after performance stabilizes: matched 192K INT8/KVarN, 64 needles,
    fixed seeds `0..9`, greedy generation, thinking/speculation/prefix reuse disabled. Report each
    `correct/64` and the arithmetic mean across ten seeds (640 retrievals per format).

## Evidence And Completion

- Exact codecs use exact bit checks; floating-point attention uses the independent represented-cache
  oracle and an explicit tolerance. Pairwise implementation parity is supplementary only.
- Distinguish the paper's experimental profile and variance-spread pseudocode from the released
  K4V2-G64 standard-deviation-spread implementation. The alignment and intentional precision/lifecycle
  differences are recorded in `docs/maintainer/paged-kv-cache.md`; do not claim paper quality results
  or byte-identical cross-engine encoding from mathematical/packed-layout agreement.
- Never claim bandwidth-, compute-, or occupancy-bound behavior without profiler evidence.
- Never call the server production-ready from microbenchmarks alone.
- Keep BF16/INT8 observable behavior and source ownership unchanged; run their focused regression
  whenever shared infrastructure changes.
- Before completion, inspect `git status`, `git diff`, and `git diff --check`. State tests run,
  baseline/new measurements, profiler attribution, quality evidence, and any unrun real-artifact
  checks. Do not commit unless requested.
