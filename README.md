# Chess-Engine-Analysis

## Setup

```bash
/usr/local/python/3.13.5/bin/python3 -m pip install --user -r requirements.txt
```

Use this full path because the default `python3` is a virtualenv that rejects
`--user`. Home directories are shared, so each person only needs to install once.

Run scripts from the repository root with `python -m`, for example
`python -m pytest tests`, so that `chesszero` can be imported.

Stockfish is needed for `chesszero/players.py` and `evaluation/move_analysis.py`.

## Shared disk

The team's shared directory is `/s/bach/a/class/cs435/cs435a/chess-zero`:

| Directory | Contents |
| --- | --- |
| `data/` | Source PGN files and other raw data |
| `checkpoints/` | Model checkpoints |
| `results/` | Match results, tables, and plots |
| `engines/` | Stockfish and other engine binaries |

The directory is shared with the `cs435a` group and must keep `g+rwx`. A default
ACL gives anything created inside it group access automatically. Files moved in
with `mv` keep their old permissions, so use `cp`, or fix them afterwards:

```bash
chmod -R g+rwX /s/bach/a/class/cs435/cs435a/chess-zero/<path>
```

## Machines

Run GPU jobs on the fish machines. `training/scan_gpus.sh` lists them, least-used
GPU first.

## Stages

1. `pipeline/`: chunk, filter, and convert PGN games to Parquet with Spark.
2. `experiments/pilot.py`: measure throughput and set the budget.
3. `training/run_condition.py`: run one condition from `configs/`.
4. `evaluation/`: run matches, Stockfish scoring, puzzles, and opening analysis.
5. `analysis/plots.py`: produce the final plots.

## File formats

To be filled in as the pipeline is written.
