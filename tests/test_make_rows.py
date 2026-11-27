"""Tests for pipeline/make_rows.py's per-game logic (no Spark needed).

Run from the repo root: python -m pytest tests/test_make_rows.py -v
"""

import random

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

from chesszero.dataset import ParquetPositions
from pipeline.make_rows import ROW_COLUMNS, base_seconds, game_to_rows, is_validation
from tests.test_dataset import game_to_rows as reference_game_to_rows
from tests.test_dataset import random_game


def pgn_from_moves(moves, result="1-0", clocks=None):
    """PGN text for a game, optionally with a [%clk] comment (seconds left) after each move."""
    game = chess.pgn.Game()
    game.headers["Result"] = result
    node = game
    for i, move in enumerate(moves):
        node = node.add_variation(move)
        if clocks is not None:
            node.set_clock(clocks[i])
    return str(game)


def rows_for(moves, result="1-0", clocks=None, time_control="600+0", min_clock=30):
    return game_to_rows("g1", pgn_from_moves(moves, result, clocks), 1500, 1500,
                        result, time_control, min_clock)


def test_matches_dataset_reference():
    """Without clocks, the rows match the reference in tests/test_dataset.py exactly."""
    rng = random.Random(0)
    for result in ("1-0", "0-1", "1/2-1/2"):
        moves = random_game(rng)
        rows, outcome, low_clock = rows_for(moves, result)
        assert outcome == "kept" and low_clock == 0
        assert rows == reference_game_to_rows("g1", moves, result)


def test_columns_and_z_signs():
    moves = [chess.Move.from_uci(u) for u in ("e2e4", "e7e5", "g1f3", "b8c6")]
    rows, _, _ = rows_for(moves, "0-1")
    assert all(tuple(r) == ROW_COLUMNS for r in rows)
    assert [r["z"] for r in rows] == [-1, 1, -1, 1]  # Black won: bad for White to move, good for Black
    assert [r["ply"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["fen"] == chess.STARTING_FEN


def test_low_clock_positions_are_skipped():
    """White drops below 30 s after their second move, so White's later positions go."""
    moves = [chess.Move.from_uci(u) for u in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3")]
    clocks = [100, 100, 20, 90, 15, 80, 10]  # seconds left after each move
    rows, outcome, low_clock = rows_for(moves, clocks=clocks)
    assert outcome == "kept"
    # Plies 0 and 2 are White with 600 and 100 s; plies 4 and 6 are White with 20 and 15 s.
    assert [r["ply"] for r in rows] == [0, 1, 2, 3, 5]
    assert low_clock == 2


def test_first_move_uses_base_time():
    moves = [chess.Move.from_uci(u) for u in ("e2e4", "e7e5")]
    rows, _, low_clock = rows_for(moves, clocks=[20, 20], time_control="20+0")
    assert rows == [] and low_clock == 2


def test_repetition_count():
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
    rows, _, _ = rows_for([chess.Move.from_uci(u) for u in shuffle], "1/2-1/2")
    assert [r["repetition_count"] for r in rows] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_skipped_games():
    moves = [chess.Move.from_uci("e2e4")]
    assert rows_for(moves, result="*")[1] == "bad_result"
    assert rows_for(moves, time_control="-")[1] == "bad_time_control"
    assert game_to_rows("g1", "1. e2e5 *", 1500, 1500, "1-0", "600+0", 30)[1] == "unreadable_pgn"
    assert rows_for([])[1] == "no_moves"


def test_base_seconds():
    assert base_seconds("600+5") == 600
    assert base_seconds("300+0") == 300
    assert base_seconds("-") is None
    assert base_seconds(None) is None


def test_validation_split_is_stable_and_about_ten_percent():
    ids = [f"game{i:06d}" for i in range(20_000)]
    picks = [is_validation(g, 0.1) for g in ids]
    assert picks == [is_validation(g, 0.1) for g in ids]
    assert 0.09 < sum(picks) / len(ids) < 0.11
    assert not any(is_validation(g, 0.0) for g in ids[:100])


def test_rows_load_in_dataset(tmp_path):
    """Rows written as Parquet load and encode through chesszero/dataset.py."""
    rng = random.Random(1)
    rows = []
    for i in range(20):
        moves = random_game(rng, max_plies=60)
        rows += game_to_rows(f"g{i}", pgn_from_moves(moves, "1/2-1/2"), 1500, 1600,
                             "1/2-1/2", "600+0", 30)[0]
    split = tmp_path / "train"
    split.mkdir()
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in ROW_COLUMNS}), split / "part-00000.parquet")
    samples = list(ParquetPositions(split, shuffle=False))
    assert len(samples) == len(rows)
    planes, move_index, mask, z = samples[0]
    assert planes.shape == (21, 8, 8) and mask[move_index]
