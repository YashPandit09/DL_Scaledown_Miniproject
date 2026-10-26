# Regeneration log

What was regenerated from the seed and compared with the saved results (T44, T46). The comparison is made by `python run.py check-regeneration`: tables must be exactly equal outside their timing columns, models and the dataset are compared by hash, and figures byte by byte.

## 1. Fresh clone (9 October 2026)

A clone of the repository from GitHub at commit `e749021`, with no `data/` and no `runs/` folder, on the project's laptop.

| Step | Result |
|---|---|
| `python run.py test` | 400 tests pass |
| `python run.py generate --width 5 --depth 4 --door W --items sofa,tv_unit,coffee_table` | works from the committed checkpoints: 62 of 64 candidates valid, 1.9 s, top 3 with quality 0.91, 0.91, 0.90 |
| `python run.py figures` | 21 figures redrawn from the saved tables and logs; `git status` shows no change, so they are byte-identical |
| `python run.py data` | dataset v1 rebuilt in about 5 minutes; content hash `ef1535ee101e7e8001c0098ce1c5d2e432686933f9dfbd05a7c20f36a7878f83`, identical to the recorded one |
| `python run.py check-regeneration` | all 20 tables equal, the dataset identical, the 4 committed checkpoints match their hashes: "everything compared is reproduced exactly" |

A second clone at commit `d84fd1e`, the code the regeneration of Section 3 ended on, passes all 412 tests.

## 2. Checks made while building

| What | Result |
|---|---|
| The three frozen CVAE seeds, retrained from `configs/frozen.yaml` | bit-identical to the three MAE runs of the Gate 2 comparison (seed 0 also to the screening run trained three days earlier) |
| B1, B2 and G0 in E1 | their rows equal the Week 1 table `baselines.csv` in every column that is not a timing |
| The screening tables E2 to E7, rebuilt from the saved runs | every column they share with the saved tables is identical; each run's M1 sampling check, repeated on the GPU, gives exactly the stored raw valid rate |
| M2 with zero optimization steps | equals M1 (same draws), tested |

One thing does *not* carry over between devices: the same seed gives different random draws on the CPU and on the GPU, so a sampling check must be repeated on the device it first ran on.

## 3. Full regeneration into an empty folder (9 October 2026)

Every step of `python run.py all --list` was run again on the project's laptop, on mains power. `SPACEGEN_OUTPUT` pointed to an empty folder outside the repository, so no dataset, trained model or cached row was reused and the saved results were not touched. It ran from 07:42 to 10:08. `python run.py check-regeneration` then compared the folder with the repository:

| What | Result |
|---|---|
| Result tables | all 21 equal outside their timing columns |
| Trained models | all 64 identical by SHA-256: the default CVAE, the 48 screening runs, the 11 runs the Gate 2 step trains, the 3 frozen seeds and the evaluator |
| Dataset v1 | identical content hash |
| Frozen configuration | the Gate 2 step chose the MAE position loss again; the `frozen.yaml` it wrote equals the repository's |
| Figures | 24 of 28 identical byte for byte; the other 4 plot a measured time and were expected to change |
| Committed checkpoints | 4 of 4 match their hashes |
| Last line of the check | `everything compared is reproduced exactly` |

Outside the check, the 48 per-epoch training logs kept in `reports/logs/screen/` were compared too: all are equal outside their `seconds` column.

**How it was run.** Not as one command. Steps that do not depend on one another ran side by side, which changes no result, because every step is seeded and deterministic by itself:

| Time | Steps |
|---|---|
| 07:42 to 08:04 | the dataset; then the baselines, the default CVAE and the evaluator together |
| 07:48 to 09:40 | the screening (48 training runs, E2 to E8), alongside everything below |
| 08:04 to 08:27 | the first passes of E1 and E10; the generator's report |
| 08:10 to 08:17 | the three frozen seeds, from the repository's `configs/frozen.yaml` |
| 08:17 to 09:24 | the final E1 and E8, the three held-out sets of E10, E12, the failure analysis; then the E10 table |
| 08:28 to 10:08 | the Gate 2 comparison (11 more training runs), then `figures --tables` |

The frozen seeds were trained from the repository's `configs/frozen.yaml` before the Gate 2 step had finished. That is sound only because the check then confirmed that Gate 2 arrives at the same file.

**The code moved while it ran.** The regeneration started at commit `185e885` and the check was run at `d84fd1e`. The commits in between change the check, the app, the PDF builder, the moment the headline script reads its speed record, and which copy of a table a figure is drawn from. None changes what a step computes, and the comparison shows it: steps run before and after those commits both reproduce the saved results.

**What the run found.**

- *A figure that depended on which command drew it.* After the screening, E5's scatter plot differed from the saved file by eight bytes although its table was identical. The screening had drawn it from unrounded numbers, while `python run.py figures` draws from the saved table, rounded to five decimals. Every experiment now reads its table back before drawing, and the `figures` step gave the identical file.
- *A figure no step regenerated.* The generator's report (a Week 1 diagnostic) was not part of `all`, although its figure is among the saved ones. It is now the second step, and its figure came out identical.
- *A check that could pass by omission.* The check looped over the regenerated tables only, so a saved table that no step had rebuilt would have gone unnoticed. It now names such tables, and compares the figures as well.

**Timings.** Up to nine processes ran at once, so the regenerated timing columns are not comparable with the saved ones, which were measured with nothing else running, and they were not copied into the repository. They do show how far a ratio can move: under this load every method's cost per valid layout rose by a quarter to a half, and M2 cost 7.6 times as much as G0 per valid layout, against 8.7 times in the saved table.

To repeat it (PowerShell; any empty folder will do):

```
$env:SPACEGEN_OUTPUT = "C:\spacegen-regeneration"
python run.py all                  # about eight hours in one go; --from STEP --to STEP runs a part
python run.py check-regeneration   # must end with "everything compared is reproduced exactly"
```

`python run.py all --list` prints the nineteen steps. After `data`, the steps `generator-report`, `baselines`, `evaluator`, `cvae`, `screening` and `frozen-seeds` do not depend on one another. After `frozen-seeds` and `evaluator`, neither do `e1`, `e8`, the three `e10-...` steps, `e12` and `failures`. Run `first-pass-e1` after `cvae` and `evaluator`, `first-pass-e10` after it, `e10` after `e1` and the three `e10-...` steps, `gate2` after `screening`, and `figures` last.

Start a step only once the models it loads exist in the output folder, because a missing model is replaced without an error: without the frozen seeds, E12 runs on the default CVAE and the failure analysis on the committed checkpoint, and without the evaluator the headline script ranks the top 3 by the rule score.

## 4. What cannot be reproduced

Timings. The laptop's speed changes with its power and thermal state, and with what else is running: the same baselines ran twice as fast one hour apart. Timing columns are left out of every comparison, and within a table all methods are timed in one pass, taking turns room by room, so that the ratios between methods hold better than the absolute times.
