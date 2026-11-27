"""Tests for chesszero/dataset.py, using small fake shards built from random games.

game_to_rows below is a reference for what pipeline/make_rows.py must produce
per game: one row per position before each move, z from the side to move's
view, and the repetition count from replaying the game.
"""

import random
from collections import Counter

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F

from chesszero.dataset import REQUIRED_COLUMNS, ParquetPositions, encode_row, make_loader
from chesszero.encoding import INPUT_PLANES, POLICY_SIZE, board_to_planes, index_to_move
from chesszero.model import ChessNet

RESULT_FOR_WHITE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def game_to_rows(game_id: str, moves: list[chess.Move], result: str) -> list[dict]:
    """Expand one game into position rows, the way make_rows.py should."""
    white_z = RESULT_FOR_WHITE[result]
    board = chess.Board()
    seen = Counter()
    rows = []
    for ply, move in enumerate(moves):
        key = " ".join(board.fen().split()[:4])
        rows.append({
            "game_id": game_id,
            "ply": ply,
            "fen": board.fen(),
            "uci": move.uci(),
            "z": white_z if board.turn == chess.WHITE else -white_z,
            "repetition_count": seen[key],
            "white_elo": 1500,
            "black_elo": 1500,
        })
        seen[key] += 1
        board.push(move)
    return rows


def random_game(rng: random.Random, max_plies: int = 120) -> list[chess.Move]:
    board = chess.Board()
    moves = []
    while len(moves) < max_plies and not board.is_game_over():
        move = rng.choice(list(board.legal_moves))
        moves.append(move)
        board.push(move)
    return moves


def write_shards(directory, rows: list[dict], n_shards: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        part = rows[i::n_shards]
        table = pa.table({name: [r[name] for r in part] for name in part[0]})
        pq.write_table(table, directory / f"part-{i:05d}.parquet")
    (directory / "_SUCCESS").touch()  # Spark writes one; it must be ignored


@pytest.fixture(scope="module")
def rows():
    rng = random.Random(0)
    out = []
    for g in range(30):
        out += game_to_rows(f"game{g}", random_game(rng), rng.choice(list(RESULT_FOR_WHITE)))
    random.Random(1).shuffle(out)  # Spark shuffles globally before writing
    return out


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory, rows):
    directory = tmp_path_factory.mktemp("rows") / "train"
    write_shards(directory, rows, n_shards=6)
    return directory


def key(row) -> tuple:
    return tuple(row[c] for c in REQUIRED_COLUMNS) if isinstance(row, dict) else tuple(row)


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_every_row_once_per_pass(shard_dir, rows, shuffle, num_workers):
    ds = ParquetPositions(shard_dir, shuffle=shuffle, shuffle_buffer=100)
    for epoch in range(2):
        ds.set_epoch(epoch)
        read = [key(r) for w in range(num_workers) for r in ds.iter_rows(w, num_workers)]
        assert Counter(read) == Counter(key(r) for r in rows)


def test_pass_is_reproducible_and_epochs_differ(shard_dir):
    ds = ParquetPositions(shard_dir, shuffle_buffer=100, seed=7)
    ds.set_epoch(0)
    first = list(ds.iter_rows())
    assert list(ds.iter_rows()) == first
    ds.set_epoch(1)
    assert list(ds.iter_rows()) != first


def test_shuffle_mixes_rows(shard_dir):
    """Consecutive rows shouldn't come from the shard in file order."""
    unshuffled = list(ParquetPositions(shard_dir, shuffle=False).iter_rows())
    shuffled = list(ParquetPositions(shard_dir, shuffle_buffer=500).iter_rows())
    assert shuffled != unshuffled
    assert shuffled[:50] != unshuffled[:50]


def test_encoded_rows_match_the_live_game():
    """What the learner sees from a row equals the live board MCTS would encode."""
    rng = random.Random(2)
    for g in range(20):
        moves = random_game(rng, max_plies=200)
        board = chess.Board()
        for row in game_to_rows(f"g{g}", moves, "1-0"):
            planes, move_index, mask, z = encode_row(row["fen"], row["uci"], row["z"], row["repetition_count"])
            assert torch.equal(planes, torch.from_numpy(board_to_planes(board)))
            assert index_to_move(int(move_index), board).uci() == row["uci"]
            assert mask[move_index]
            assert mask.sum() == board.legal_moves.count()
            assert z == (1 if board.turn == chess.WHITE else -1)
            board.push_uci(row["uci"])


def test_z_is_not_flipped_for_black():
    """z is stored from the side to move's view; the dataset passes it through."""
    board = chess.Board()
    board.push_uci("e2e4")
    _, _, _, z = encode_row(board.fen(), "e7e5", -1, 0)
    assert z == -1.0


def test_corrupt_row_raises():
    with pytest.raises(ValueError):
        encode_row(chess.STARTING_FEN, "e7e5", 0, 0)  # Black's move with White to move


def test_missing_column_raises(tmp_path):
    pq.write_table(pa.table({"fen": [chess.STARTING_FEN], "uci": ["e2e4"], "z": [0]}), tmp_path / "a.parquet")
    with pytest.raises(ValueError, match="repetition_count"):
        ParquetPositions(tmp_path)


def test_no_shards_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ParquetPositions(tmp_path)


def test_dataloader_feeds_the_model(shard_dir, rows):
    """End to end: workers, batching, and a masked loss on the stand-in network."""
    ds = ParquetPositions(shard_dir, shuffle_buffer=100)
    loader = make_loader(ds, batch_size=64, num_workers=2)
    net = ChessNet(2, 16, tower="stand_in")
    n = 0
    for planes, move_index, mask, z in loader:
        assert planes.shape[1:] == (INPUT_PLANES, 8, 8) and planes.dtype == torch.float32
        assert mask.shape[1:] == (POLICY_SIZE,) and mask.dtype == torch.bool
        assert move_index.dtype == torch.int64 and z.dtype == torch.float32
        assert mask[torch.arange(len(move_index)), move_index].all()
        policy, value = net(planes)
        loss = F.cross_entropy(policy.masked_fill(~mask, float("-inf")), move_index) + F.mse_loss(value.squeeze(1), z)
        assert torch.isfinite(loss)
        n += len(z)
    assert n == len(rows)


def test_set_epoch_reaches_workers(shard_dir):
    """Workers get a fresh copy of the dataset each pass, so a new epoch reorders rows."""
    ds = ParquetPositions(shard_dir, shuffle_buffer=100)
    loader = make_loader(ds, batch_size=64, num_workers=2)
    orders = []
    for epoch in (0, 0, 1):
        ds.set_epoch(epoch)
        orders.append(torch.cat([move_index for _, move_index, _, _ in loader]))
    assert torch.equal(orders[0], orders[1])
    assert not torch.equal(orders[0], orders[2])
