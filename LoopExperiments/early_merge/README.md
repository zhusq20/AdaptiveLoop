# Early state merge pilot

Requested device: physical CUDA 0. Backbone: the existing local Ouro-2.6B checkpoint.

This directory tests a **compact, one-state-per-branch interface**. Each worker
independently solves a one-operation integer arithmetic problem. The receiving
Ouro model sees only a fixed program that combines two supplied values, not the
workers' input programs. The source state is the last prompt position immediately
before its single-digit answer. Both branches share the same frozen backbone.

## Protocol fixed before collection

- Values and answers are digits 0–9. Workers use `x +/- c`, with nonzero `c`, and
  exactly nine distinct expressions per result. Six expressions/result form the
  interface-training split, three form the held-out evaluation split (seed 2718).
- The receiver computes `a + b` or `a - b`. Every evaluated result remains 0–9.
  Correct digit embeddings at its two input slots are the text-oracle control.
- Collect source states and unrestricted vocabulary predictions at depths 0–4.
  Primary accuracy uses the unrestricted prediction, not argmax restricted to
  digits. Restricted-digit accuracy is a diagnostic.
- First evaluate the receiver on correct textual inputs and evaluate the workers
  independently. Then test raw source-state injection and RMS-matched injection
  at the receiver's embedding boundary, with 1–4 receiver loops. Source depth and
  receiver depth are separate variables. The latter are **new receiver passes**,
  not an assertion that all computation fits within four total loops.
- A source hidden state predicts a digit, whereas a receiver operand normally
  consumes a digit embedding. Native injection is therefore a substantive
  interface test. Failure does not refute early latent composition in general.
- If native injection fails despite competent workers and a competent text
  receiver, try a small shared adapter with a frozen backbone. Clearly distinguish
  any supervised representation calibration from untrained model capability.
- Evaluate independent left/right replacement and double replacement. Report all
  items, not only examples whose original answer was correct. Match branch depths,
  receiver budgets, and communication width across interventions. Include depth 0
  and equivalent-result/different-expression checks.
- Source branches never see each other's inputs or the final merge operation.
  The receiver never sees their raw expressions. Hidden states may still encode
  raw operands; successful transfer alone does not prove abstract subanswer use.
- No acceleration claim from accuracy alone. Record wall time and state explicitly
  whether it includes extraction, warmup, training, and state transport.

Progress and results will be recorded with the run artifacts. Existing `dag_depth`
experiments and other GPUs are not modified.

## Execution decisions (2026-11-12)

The first one-operation control was perfect at every worker and text-receiver
depth. Direct raw/rms-normalized final-state injection failed, so it cannot test
the proposed early-readout gap. A two-operation calibration set had a substantial
depth-1 to depth-4 improvement; a three-operation calibration set was only 54%
correct at depth 4. These calibration runs selected the two-operation task before
the main interface training and evaluation.

The main run (`runs/ops2_full_2718`) exhaustively covers all 810 legal two-operation
programs of the form `x +/- c1 +/- c2`, where constants are nonzero digits and all
intermediate results stay in 0–9. It uses 450 source programs for training, 90 for
validation, and 270 for test, with every result balanced in each split. The final
400 test quartets are balanced by base answer; the other counterfactual labels are
not necessarily balanced. Each quartet uses three distinct equivalent-result
source routes. All four counterfactual answers within a quartet are distinct.

The shared adapter has 526,464 trainable parameters (LayerNorm without affine
parameters; 2048→128→2048 GELU mapper plus a constant embedding offset). Both workers
and the receiver are frozen Ouro-2.6B. Only the final answer cross-entropy trains
the adapter: no subanswer loss, source-program text, opposite-branch information,
or merge-operation input is supplied to it. Training is fixed at 400 AdamW steps,
batch 48, learning rate .001; first 100 steps use depth 4, subsequent steps mix
depths 0–4. The receiver uses one loop during training. Checkpoint selection uses
mean validation accuracy at source depths 1–4, only after mixed-depth training
begins. Held-out depths and extra receiver-loop budgets use the same checkpoint.

An independently supervised ridge readout predicts each worker's digit. Its
regularization is selected on validation programs only; this explicitly measures
the possibility that a latent merge advantage is a readout advantage.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python early_merge/pilot.py --stage collect --source-ops 2 --per-value 81 --n 400 --out early_merge/runs/ops2_full_2718
CUDA_VISIBLE_DEVICES=0 .venv/bin/python early_merge/adapter.py --run early_merge/runs/ops2_full_2718
.venv/bin/python early_merge/probe.py --run early_merge/runs/ops2_full_2718
.venv/bin/python early_merge/analyze.py --run early_merge/runs/ops2_full_2718
CUDA_VISIBLE_DEVICES=0 .venv/bin/python early_merge/baselines.py --run early_merge/runs/ops2_full_2718
```

The scripts save per-example predictions and selected checkpoint metadata. The
primary latent accuracy is the balanced base-case accuracy at one receiver loop;
counterfactual all-four success requires all four distinct outputs to be right.
The confidence intervals resample whole base quartets, retaining all routes and
swaps, and are conditional on the fixed source pool and one training seed.

The latency benchmark uses one request and two workers batched on CUDA 0. It
includes complete worker/receiver forward passes and the adapter, and excludes
CPU tokenization and network transport. Text handoff is optimistically implemented
as an on-device digit lookup. FLOPs are dense mathematical estimates, not hardware
counters. The serial baseline uses the frozen model without the adapter's training;
loops 5–8 are extrapolation beyond its four-loop pretraining schedule.
