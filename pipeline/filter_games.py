"""Header filters plus counts after each filter.

Reads the month's PGN chunks (pipeline/chunk_pgn.py) and writes one row per kept
game as Parquet: game_id, white_elo, black_elo, result, time_control, pgn. That's
the input to pipeline/make_rows.py.

A game is dropped by the first rule it fails, in this order (docs/decisions.md #9):

    bot                      either player has the BOT title
    not_normal_termination   Termination isn't "Normal" (time forfeits, abandoned,
                             rules infractions, unterminated)
    short_time               base time below data.filter_min_base_seconds, or no
                             time control ("-")
    low_rating               either rating below data.filter_min_rating, or missing
    bad_header               a header line that can't be parsed

Usage (from the repo root):

    spark-submit pipeline/filter_games.py <chunks_dir> <output_dir>
    python pipeline/filter_games.py <one_chunk.pgn.zst>     # counts for one chunk, no Spark

Paths on the shared folder need a file:// prefix, or Spark reads them from HDFS.
<output_dir>/_stats.json gets the settings and the count for each rule. (The leading
underscore makes Spark skip it when make_rows.py reads the folder as Parquet.)

Written by Munisa Yakubova; finished by Ryan Baldwin (single pass, Normal-only
termination, thresholds from configs/base.yaml, stats.json).
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

import zstandard

RULES = ("bot", "not_normal_termination", "short_time", "low_rating", "bad_header")


def split_games(data):
    """Return the list of games in one compressed chunk (given as bytes)."""
    raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data), read_across_frames=True)
    text = io.TextIOWrapper(raw, encoding="utf-8").read()
    parts = text.split("\n\n[Event ")
    return [parts[0]] + ["[Event " + p for p in parts[1:]]


def parse_headers(game):
    """Return the header lines of one game as a dictionary, or None if one can't be parsed."""
    headers = {}
    for line in game.splitlines():
        if line.startswith("["):
            if not line.endswith("]") or " " not in line:
                return None
            key, value = line[1:-1].split(" ", 1)
            headers[key] = value.strip('"')
    return headers


def check_rules(headers, min_base_seconds, min_rating):
    """Return the name of the first rule the game fails, or 'kept' if it passes all."""
    if headers is None:
        return "bad_header"
    if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
        return "bot"
    if headers.get("Termination") != "Normal":
        return "not_normal_termination"
    base = headers.get("TimeControl", "-").split("+")[0]
    if not base.isdigit() or int(base) < min_base_seconds:
        return "short_time"
    for key in ("WhiteElo", "BlackElo"):
        rating = headers.get(key, "?")
        if not rating.isdigit() or int(rating) < min_rating:
            return "low_rating"
    return "kept"


def to_row(game, headers):
    """Return the columns saved for one kept game."""
    game_id = headers.get("Site", "").split("/")[-1]
    return (game_id, int(headers["WhiteElo"]), int(headers["BlackElo"]),
            headers.get("Result"), headers.get("TimeControl"), game)


def run_spark(in_dir, out_dir, data_cfg):
    from pyspark.sql import SparkSession

    min_base = data_cfg["filter_min_base_seconds"]
    min_rating = data_cfg["filter_min_rating"]
    spark = SparkSession.builder.appName("filter_games").getOrCreate()
    sc = spark.sparkContext
    # One pass: the write is the only action over the games, and the accumulators
    # count each rule along the way. If Spark re-runs a failed task, its games
    # would be counted twice, so the kept total in stats.json is read back from
    # the output instead.
    counts = {name: sc.accumulator(0) for name in RULES + ("kept", "games_in")}

    def keep(game):
        headers = parse_headers(game)
        rule = check_rules(headers, min_base, min_rating)
        counts["games_in"].add(1)
        counts[rule].add(1)
        return [to_row(game, headers)] if rule == "kept" else []

    chunks = sc.binaryFiles(in_dir, minPartitions=1000)
    rows = chunks.flatMap(lambda pair: split_games(pair[1])).flatMap(keep)
    schema = "game_id string, white_elo int, black_elo int, result string, time_control string, pgn string"
    out = out_dir.rstrip("/")
    spark.createDataFrame(rows, schema).write.mode("overwrite").parquet(out)

    stats = {"settings": {"filter_min_base_seconds": min_base, "filter_min_rating": min_rating},
             "input": in_dir,
             "counts_by_first_failed_rule": {k: v.value for k, v in counts.items()},
             "kept_games_written": spark.read.parquet(out).count()}
    text = json.dumps(stats, indent=2)
    if out.startswith("file://"):
        Path(out[len("file://"):], "_stats.json").write_text(text + "\n", encoding="utf-8")
    else:
        sc.parallelize([text], 1).saveAsTextFile(f"{out}/_stats.json")
    spark.stop()
    return stats


def main(argv):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from chesszero.config import load_config

    data_cfg = load_config()["data"]
    if len(argv) == 2:
        games = split_games(open(argv[1], "rb").read())
        print(len(games), "games")
        print(Counter(check_rules(parse_headers(g), data_cfg["filter_min_base_seconds"],
                                  data_cfg["filter_min_rating"]) for g in games))
    elif len(argv) == 3:
        print(json.dumps(run_spark(argv[1], argv[2], data_cfg), indent=2))
    else:
        sys.exit("usage: spark-submit pipeline/filter_games.py <chunks_dir> <output_dir>")


if __name__ == "__main__":
    main(sys.argv)
