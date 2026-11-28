"""Summarize the human-game data for the write-up's figures.

Reads the filter output and the validation rows on the shared folder and writes
small histograms to a JSON file, which analysis/pipeline_figures.ipynb plots. Run it once
per month of data (a few minutes; it needs a few GB of memory, not Spark):

    python -m analysis.data_stats                       # October 2026
    python -m analysis.data_stats --month 2026-10 --out results/data_stats_2026-10.json

Sources:
- filtered_<month>/  (every game that passed filter_games.py): ratings, time
  controls and results, split by whether make_rows.py used the game
  (base >= data.rows_min_base_seconds).
- rows_<month>/val/  (a 10% sample of the row games, chosen by hash of game ID, so
  it holds whole games): rows per game and the ply of each row. The train split
  is the same distribution and nine times larger, so it isn't read.
- the _stats.json / stats.json counts the two Spark jobs wrote.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from chesszero.config import load_config
from pipeline.make_rows import base_seconds

DATA = Path("/s/bach/a/class/cs435/cs435a/chess-zero/data")
RATING_BINS = np.arange(1000, 3301, 50)  # filter keeps both ratings >= 1000
RESULT_RATING_BINS = np.arange(1000, 2801, 100)  # coarser bins for results by rating
MAX_PLY = 400                           # histograms put longer games in the last bin
RESULTS = ("1-0", "1/2-1/2", "0-1")


def filtered_summary(directory: Path, rows_min_base: int) -> dict:
    table = ds.dataset(directory, format="parquet").to_table(
        columns=["white_elo", "black_elo", "result", "time_control"])
    tc = table.column("time_control").to_pylist()
    in_rows = np.array([(base_seconds(t) or 0) >= rows_min_base for t in tc])
    mean_elo = (table.column("white_elo").to_numpy() + table.column("black_elo").to_numpy()) / 2
    results = np.array(table.column("result").to_pylist())

    out = {"games": len(tc), "rating_bins": RATING_BINS.tolist(),
           "result_rating_bins": RESULT_RATING_BINS.tolist()}
    for name, mask in (("in_rows", in_rows), ("not_in_rows", ~in_rows)):
        out[name] = {
            "games": int(mask.sum()),
            "mean_rating_hist": np.histogram(np.clip(mean_elo[mask], 1000, 3299), RATING_BINS)[0].tolist(),
            "results": dict(Counter(results[mask].tolist())),
            # games per 100-point bin of average rating, for each result (last bin holds 2800+)
            "results_by_rating": {
                r: np.histogram(np.clip(mean_elo[mask & (results == r)], 1000, 2799),
                                RESULT_RATING_BINS)[0].tolist() for r in RESULTS},
        }
    out["time_controls"] = dict(Counter(tc).most_common())
    return out


def val_rows_summary(directory: Path) -> dict:
    table = ds.dataset(directory, format="parquet").to_table(columns=["game_id", "ply", "z"])
    ply = table.column("ply").to_numpy()
    # z is from the side to move's view; flip it on Black's plies to get the result for White.
    white_z = pc.multiply(table.column("z"), pc.subtract(1, pc.multiply(2, pc.bit_wise_and(table.column("ply"), 1))))
    games = table.append_column("white_z", white_z).group_by("game_id").aggregate(
        [("ply", "count"), ("white_z", "max")])
    per_game = games.column("ply_count").to_numpy()
    game_result = games.column("white_z_max").to_numpy()
    z = pc.value_counts(table.column("z")).to_pylist()
    return {
        "rows": len(ply),
        "games": len(per_game),
        "rows_per_game_hist": np.bincount(np.minimum(per_game, MAX_PLY), minlength=MAX_PLY + 1).tolist(),
        "rows_per_game_hist_by_result": {
            r: np.bincount(np.minimum(per_game[game_result == wz], MAX_PLY), minlength=MAX_PLY + 1).tolist()
            for r, wz in zip(RESULTS, (1, 0, -1))},
        "rows_by_ply": np.bincount(np.minimum(ply, MAX_PLY), minlength=MAX_PLY + 1).tolist(),
        "rows_per_game_median": float(np.median(per_game)),
        "rows_per_game_mean": float(per_game.mean()),
        "z_counts": {str(d["values"]): d["counts"] for d in z},
        "max_ply_bin_note": f"bin {MAX_PLY} holds every value >= {MAX_PLY}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", default="2026-10")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config()["data"]
    out_path = Path(args.out or f"results/data_stats_{args.month}.json")

    filtered = DATA / f"filtered_{args.month}"
    rows = DATA / f"rows_{args.month}"
    summary = {
        "month": args.month,
        "rows_min_base_seconds": cfg["rows_min_base_seconds"],
        "filter_stats": json.loads((filtered / "_stats.json").read_text()),
        "rows_stats": json.loads((rows / "stats.json").read_text()),
        "filtered": filtered_summary(filtered, cfg["rows_min_base_seconds"]),
        "val_rows": val_rows_summary(rows / "val"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
