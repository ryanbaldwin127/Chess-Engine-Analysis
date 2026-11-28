"""Header filters plus counts after each filter."""
import io
import sys
from collections import Counter
import zstandard

MIN_BASE_SECONDS = 300
MIN_RATING = 1200


def split_games(data):
    """Return the list of games in one compressed chunk (given as bytes)."""
    raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data), read_across_frames=True)
    text = io.TextIOWrapper(raw, encoding="utf-8").read()
    parts = text.split("\n\n[Event ")
    return [parts[0]] + ["[Event " + p for p in parts[1:]]


def parse_headers(game):
    """Return the header lines of one game as a dictionary."""
    headers = {}
    for line in game.splitlines():
        if line.startswith("["):
            key, value = line[1:-1].split(" ", 1)
            headers[key] = value.strip('"')
    return headers


def check_rules(headers):
    """Return the name of the first rule the game fails, or 'kept' if it passes all."""
    if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
        return "bot"
    if headers.get("Termination") == "Time forfeit":
        return "time_forfeit"
    base = headers.get("TimeControl", "-").split("+")[0]
    if not base.isdigit() or int(base) <= MIN_BASE_SECONDS:
        return "short_time"
    for key in ("WhiteElo", "BlackElo"):
        rating = headers.get(key, "?")
        if not rating.isdigit() or int(rating) < MIN_RATING:
            return "low_rating"
    return "kept"


def to_row(game, headers):
    """Return the columns saved for one kept game."""
    game_id = headers.get("Site", "").split("/")[-1]
    return (game_id, int(headers["WhiteElo"]), int(headers["BlackElo"]),
            headers.get("Result"), headers.get("TimeControl"), game)


def run_spark(in_dir, out_dir):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("filter_games").getOrCreate()

    chunks = spark.sparkContext.binaryFiles(in_dir, minPartitions=1000)
    games = chunks.flatMap(lambda pair: split_games(pair[1]))
    with_headers = games.map(lambda g: (g, parse_headers(g)))

    counts = with_headers.map(lambda gh: check_rules(gh[1])).countByValue()
    print("COUNTS:", dict(counts))

    kept = with_headers.filter(lambda gh: check_rules(gh[1]) == "kept")
    rows = kept.map(lambda gh: to_row(gh[0], gh[1]))
    schema = "game_id string, white_elo int, black_elo int, result string, time_control string, pgn string"
    spark.createDataFrame(rows, schema).write.mode("overwrite").parquet(out_dir)
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        games = split_games(open(sys.argv[1], "rb").read())
        print(len(games), "games")
        print(Counter(check_rules(parse_headers(g)) for g in games))
    else:
        run_spark(sys.argv[1], sys.argv[2])
