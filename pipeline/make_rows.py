"""Games to rows, split by game ID, shuffle, write Parquet.

Reads the filter's output (pipeline/filter_games.py: one row per game with
game_id, white_elo, black_elo, result, time_control, pgn) and writes one row per
position in the format chesszero/dataset.py reads (docs/decisions.md #23):

    game_id, ply, fen, uci, z, repetition_count, white_elo, black_elo

Rules applied here:
  - Time controls: only games with a base time of at least
    data.rows_min_base_seconds (default 600 s, decision #11).
  - Low clock: a position is skipped if the player to move had less than
    data.min_clock_seconds left when they moved (decision #10). The rest of the
    game is still used, and z stays the game result.
  - Split: a game goes to validation if a hash of its ID falls in the first
    data.val_fraction of the hash range (decision #5), so all its positions are in
    one split and the split doesn't depend on Spark's partitioning.
  - Shuffle: rows are spread across data.shards shards at random and shuffled
    within each shard, so every shard mixes thousands of games. Each shard writes
    one train file and one (about ten times smaller) val file.

Usage (from the repo root):

    spark-submit pipeline/make_rows.py <filtered_games_parquet> <output_dir>

    <output_dir>/train/ and <output_dir>/val/ get the shards, and
    <output_dir>/stats.json the counts for the write-up. Paths on the shared
    folder need a file:// prefix, or Spark reads them from HDFS.

The per-game logic (game_to_rows) is plain Python, tested in
tests/test_make_rows.py without Spark.
"""

import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

RESULT_FOR_WHITE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

ROW_COLUMNS = ("game_id", "ply", "fen", "uci", "z", "repetition_count", "white_elo", "black_elo")

# Per-game outcomes counted for stats.json.
SKIP_REASONS = ("bad_time_control", "short_time_control", "bad_result", "unreadable_pgn", "no_moves")


def base_seconds(time_control: str | None) -> int | None:
    """Base time in seconds from a Lichess TimeControl like "600+5", or None if there isn't one."""
    if not time_control:
        return None
    base = time_control.split("+")[0]
    return int(base) if base.isdigit() else None


def is_validation(game_id: str, val_fraction: float) -> bool:
    """True if this game belongs to the validation split.

    Uses SHA-1 of the game ID rather than Python's hash(), which changes between
    processes, so every machine and every run agree.
    """
    digest = hashlib.sha1(game_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_fraction


def game_to_rows(game_id, pgn_text, white_elo, black_elo, result, time_control, min_clock_seconds):
    """Expand one game into position rows.

    Returns (rows, outcome, low_clock_skipped): rows is a list of dicts with
    ROW_COLUMNS, outcome is "kept" or one of SKIP_REASONS, and low_clock_skipped
    counts positions dropped by the low-clock rule.

    The player to move's clock is the time they had left after their previous
    move (Lichess's [%clk] comment, which includes any increment), or the base
    time for each side's first move. Games without clock comments keep every
    position.
    """
    white_z = RESULT_FOR_WHITE.get(result)
    if white_z is None:
        return [], "bad_result", 0
    base = base_seconds(time_control)
    if base is None:
        return [], "bad_time_control", 0

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None or game.errors:
        return [], "unreadable_pgn", 0

    board = game.board()
    seen = Counter()
    clock_left = {chess.WHITE: float(base), chess.BLACK: float(base)}
    rows = []
    low_clock = 0
    for ply, node in enumerate(game.mainline()):
        move = node.move
        key = " ".join(board.fen().split()[:4])
        if clock_left[board.turn] >= min_clock_seconds:
            rows.append({
                "game_id": game_id,
                "ply": ply,
                "fen": board.fen(),
                "uci": move.uci(),
                "z": white_z if board.turn == chess.WHITE else -white_z,
                "repetition_count": seen[key],
                "white_elo": int(white_elo),
                "black_elo": int(black_elo),
            })
        else:
            low_clock += 1
        seen[key] += 1
        clock = node.clock()
        if clock is not None:
            clock_left[board.turn] = clock
        board.push(move)

    if not rows and not low_clock:
        return [], "no_moves", 0
    return rows, "kept", low_clock


def run_spark(input_path: str, output_dir: str, data_cfg: dict) -> dict:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.appName("make_rows").getOrCreate()
    sc = spark.sparkContext
    min_base = data_cfg["rows_min_base_seconds"]
    min_clock = data_cfg["min_clock_seconds"]
    val_fraction = data_cfg["val_fraction"]
    seed = data_cfg["seed"]

    # Accumulators report why games were skipped. Spark can re-run a failed task,
    # which would count its games twice, so stats.json takes the row and game
    # totals from the written output instead, and these are for diagnosis.
    counters = {name: sc.accumulator(0) for name in ("games_in", "kept", "low_clock_positions") + SKIP_REASONS}

    def expand(game):
        counters["games_in"].add(1)
        base = base_seconds(game.time_control)
        if base is not None and base < min_base:
            counters["short_time_control"].add(1)
            return []
        rows, outcome, low_clock = game_to_rows(
            game.game_id, game.pgn, game.white_elo, game.black_elo,
            game.result, game.time_control, min_clock)
        counters[outcome].add(1)
        counters["low_clock_positions"].add(low_clock)
        split = "val" if is_validation(game.game_id, val_fraction) else "train"
        return [tuple(r[c] for c in ROW_COLUMNS) + (split,) for r in rows]

    games = spark.read.parquet(input_path)
    schema = ("game_id string, ply int, fen string, uci string, z int, repetition_count int, "
              "white_elo int, black_elo int, split string")
    rows = spark.createDataFrame(games.rdd.flatMap(expand), schema)

    # One shuffle and one write for both splits: rows go to random shards, are
    # shuffled within each shard, and partitionBy("split") writes each shard's
    # train and val rows to split=train/ and split=val/. Every game is parsed
    # once, so the accumulators count each game once (barring task retries), and
    # no copy of the rows has to be kept on the workers' scratch disks.
    out = output_dir.rstrip("/")
    (rows.repartition(data_cfg["shards"], F.rand(seed))
         .sortWithinPartitions(F.rand(seed + 1))
         .write.mode("overwrite").partitionBy("split").parquet(out))
    for split in ("train", "val"):
        _rename(spark, f"{out}/split={split}", f"{out}/{split}")

    # Exact totals and the no-overlap check (#5) come from reading the output back.
    stats = {"settings": dict(data_cfg), "input": input_path,
             "games_seen_by_tasks": {k: v.value for k, v in counters.items()}}
    ids = {}
    for split in ("train", "val"):
        written = spark.read.parquet(f"{out}/{split}")
        ids[split] = written.select("game_id").distinct()
        stats[split] = {"rows": written.count(), "games": ids[split].count()}
    stats["games_in_both_splits"] = ids["train"].join(ids["val"], "game_id").count()
    if stats["games_in_both_splits"]:
        raise RuntimeError(f"{stats['games_in_both_splits']} games appear in both train and val")

    # stats.json goes next to the splits. On the shared folder (file://) the driver
    # writes it directly; on HDFS it goes through Spark as a one-part text file.
    text = json.dumps(stats, indent=2)
    if out.startswith("file://"):
        Path(out[len("file://"):], "stats.json").write_text(text + "\n", encoding="utf-8")
    else:
        sc.parallelize([text], 1).saveAsTextFile(f"{out}/stats.json")
    spark.stop()
    return stats


def _rename(spark, src: str, dst: str) -> None:
    """Rename a directory through Hadoop's file system layer, so file:// and HDFS both work."""
    jvm = spark._jvm
    src_path = jvm.org.apache.hadoop.fs.Path(src)
    fs = src_path.getFileSystem(spark._jsc.hadoopConfiguration())
    dst_path = jvm.org.apache.hadoop.fs.Path(dst)
    if fs.exists(dst_path):
        fs.delete(dst_path, True)
    if not fs.rename(src_path, dst_path):
        raise RuntimeError(f"could not rename {src} to {dst}")


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: spark-submit pipeline/make_rows.py <filtered_games_parquet> <output_dir>")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from chesszero.config import load_config, require_complete

    cfg = load_config()
    require_complete(cfg, sections=["data"])
    stats = run_spark(argv[1], argv[2], cfg["data"])
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(sys.argv)
