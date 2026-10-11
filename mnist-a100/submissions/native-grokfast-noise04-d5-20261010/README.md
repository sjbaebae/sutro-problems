# Experimental D5 BF16 Grokfast + input noise

**Not qualified; not a leaderboard submission.** The complete recovery batch passed only one of three canonical Modal A100-80GB sandbox evaluations. All three outcomes are retained, including both accuracy failures. There is no qualifying median latency or energy. The existing qualified depth-ensemble PR115 is unchanged.

| Run | MNIST error | Status | Ranked ms | mJ per call |
|---|---:|---|---:|---:|
| 1 | 1.908182% | FAIL | unranked | not measured |
| 2 | 2.000000% | FAIL | unranked | not measured |
| 3 | 1.883636% | PASS | 3541.172 | 964709.742 |

Run 1 was 9 correct predictions short of the D5 limit; run 2 was 110 short. Both failures were accuracy only. All evaluations completed 15 calls, including 11 MNIST draws, with matching source hashes and sandbox enabled. Diagnostic MNIST warm times for the failed runs were 3532.250 and 3539.343 ms; these are unranked. Energy is normally skipped after accuracy failure. GPU UUIDs therefore cannot establish three distinct devices; three independent single-use containers were launched by the unchanged canonical runner.

## Interrupted first attempt and recovery

An earlier three-call batch was interrupted by a local `OSError(28, No space left on device)`. Modal showed two remote calls completed and one cancelled, but their benchmark outputs could not be recovered. Their accuracy is **unknown**, not assumed passing. The original manifest, infrastructure audit, and recovery error are in `interrupted-attempt/`. The user explicitly requested fixing disk space and pushing the code. Regenerable caches were cleaned, then this separately identified recovery batch used the unchanged `run_modal.remote_score` function with persistent detached call handles. This was an infrastructure recovery, not the first-ever qualification attempt. No completed recovery result was replaced or rerun.

## Learner

Two teachers, each eight width-256 three-hidden-layer BN/ReLU MLPs, train for 1000 and 250 updates. Their averaged lookup/graph probability supplies hard pseudo labels for query examples at confidence >=0.999. Four width-512, three-hidden-layer BN/ReLU students are warm-started from paired teachers and train for 3000 updates. Student input comprises 60 supplied features plus ten fixed class-mean dot similarities computed only from known labels, physically padded to 128. The new input columns initially have zero weights.

Student training uses Grokfast slow-gradient EMA alpha0.98/lambda0.5 before clipping, weight decay and Nesterov momentum0.95. Public LR is0.4/1.5 with cosine decay; the width-scaled head uses the existing lower effective LR. Dropout0.1, weight decay0.001, mixup0.2, batch2048 with two views, final-quarter SWA and residual gain0.25 remain fixed. Per-update Gaussian noise sd0.04 is applied to raw mixed student inputs; centroid similarities are recomputed consistently. This is feature-space noise, not pixel augmentation. Parameters, gradients and momentum are BF16; accumulation, normalization statistics, SWA and slow-gradient EMA are FP32.

Prediction concatenates centered, half-normalized middle/final student embeddings. Five-neighbor voting uses only the original 10000 true-label embeddings, followed by three query-graph propagation steps. The package contains source only, no fitted weights or data. CUDA/cuBLASLt kernels run in CUDA graphs; this is not a fully fused megakernel.

## Research evidence and package

The strength sweep selected sd0.04 at1.826364% over11 draws. Independent fixed-recipe validation reached1.844545%; its no-noise control was better at1.817273%, so the noise advantage did not replicate. These research results do not override the failed canonical qualification. Summaries and the cold compile/reset gate are in `research/`.

`submission.py` is20457 bytes, SHA256 `aa028894701ef993382393307af1f7c4e4dd4505b1f329b268e5272eac05b4f7`. All22 readable archived source files are in `source/`, checked against `source_manifest.json`; CUDA is stored in its own files. Packaging removes unused initialization and augmentation modes, redundant fixed-configuration checks, and diagnostic exports; it uses bounded cuBLASLt tuning and minimal ATen bindings. The packaged cold/reset/input-immutability gate passed. Scripts under `provenance-scripts/` document the research workspace workflow and may require its directory structure; canonical reproduction uses the repository runner below.

From `mnist-a100`:

```sh
python run_modal.py submissions/native-grokfast-noise04-d5-20261010/submission.py:classify --difficulty 5 --runs 3 --json results/noise04-review
```

The recovery used Torch2.12, CUDA13.3 and Python3.13. All raw calls and the independent canonical-judge audit are in `sandbox-recovery/`. No leaderboard edits or record claim are included.
