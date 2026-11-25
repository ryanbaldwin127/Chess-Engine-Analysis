"""PyTorch Dataset streaming supervised positions from Parquet shards.

This module is the contract between pipeline/make_rows.py (which writes the
shards) and the supervised learner (which reads them). Change the row format
only together with make_rows.py and tests/test_dataset.py.

Shards live on the shared folder (docs/decisions.md #7), one directory per split,
e.g. <shared>/data/rows_2026-10/train/ and .../val/. The split is by hash of
game ID, done in Spark, so a game's positions are all in one split. Every
*.parquet file in the directory is a shard; other files (Spark's _SUCCESS) are
ignored. Spark shuffles the rows globally before writing, so each shard should
already mix thousands of games.

Row format, one row per position (docs/decisions.md #23):

    column            type     meaning
    game_id           string   Lichess game ID (from filter_games.py)
    ply               int      0 for the starting position, 1 after White's first move, ...
    fen               string   the position before the move, from board.fen()
    uci               string   the move the human played, e.g. "e2e4", "e7e8n"
    z                 int      game result from the SIDE TO MOVE's view:
                               +1 the player to move went on to win, 0 draw, -1 loss
    repetition_count  int      how many times this position occurred earlier in the
                               game, keyed on the first four FEN fields (#13)
    white_elo         int      ratings, carried over for analysis
    black_elo         int

Only fen, uci, z and repetition_count are read for training (REQUIRED_COLUMNS);
the rest are for analysis and later cuts, and any extra columns are ignored.

Each sample is a tuple of tensors, batched by the default collate:

    planes      (21, 8, 8) float32   board_to_planes(board, repetition_count)
    move_index  ()         int64     move_to_index(move, board), the policy target
    mask        (4672,)    bool      legal_move_mask(board)
    z           ()         float32   value target, as stored
"""

import random
from pathlib import Path

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from chesszero.encoding import board_to_planes, legal_move_mask, move_to_index

REQUIRED_COLUMNS = ("fen", "uci", "z", "repetition_count")


def list_shards(directory: str | Path) -> list[Path]:
    """Return the Parquet shards in a split directory, sorted by name."""
    shards = sorted(Path(directory).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no *.parquet shards in {directory}")
    return shards


def encode_row(fen: str, uci: str, z: float, repetition_count: int):
    """Encode one row as (planes, move_index, mask, z) tensors.

    Raises ValueError if the move isn't legal in the position, which would mean
    the row is corrupt (e.g. a FEN and move from different plies).
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    if not board.is_legal(move):
        raise ValueError(f"row move {uci} is not legal in {fen}")
    return (
        torch.from_numpy(board_to_planes(board, repetition_count=int(repetition_count))),
        torch.tensor(move_to_index(move, board), dtype=torch.int64),
        torch.from_numpy(legal_move_mask(board)),
        torch.tensor(z, dtype=torch.float32),
    )


class ParquetPositions(IterableDataset):
    """Stream encoded positions from the shards of one split.

    One pass over the data reads every row exactly once. With shuffle=True, each
    pass reads the shards in an order drawn from (seed, epoch) and mixes rows
    through a shuffle buffer, so a pass is reproducible given seed, epoch and
    the DataLoader's num_workers. With shuffle=False (validation), rows come in
    file order.

    Shards are dealt to DataLoader workers round-robin, so no row is read twice
    in a pass; use at least as many shards as workers or some workers sit idle.

    Call set_epoch before each pass. On resume, the learner should start a new
    epoch number rather than try to skip into the interrupted pass (#23).
    """

    def __init__(
        self,
        directory: str | Path,
        shuffle: bool = True,
        shuffle_buffer: int = 10_000,
        seed: int = 0,
        read_batch_rows: int = 4096,
    ):
        self.shards = list_shards(directory)
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.read_batch_rows = read_batch_rows
        self.epoch = 0
        for shard in self.shards[:1]:
            check_schema(shard)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def shards_for_worker(self, worker_id: int, num_workers: int) -> list[Path]:
        shards = list(self.shards)
        if self.shuffle:
            random.Random(f"{self.seed}:{self.epoch}").shuffle(shards)
        return shards[worker_id::num_workers]

    def iter_rows(self, worker_id: int = 0, num_workers: int = 1):
        """Yield raw rows (fen, uci, z, repetition_count) for one worker's share of a pass."""
        rows = self._read(self.shards_for_worker(worker_id, num_workers))
        if not self.shuffle:
            yield from rows
            return
        rng = random.Random(f"{self.seed}:{self.epoch}:{worker_id}")
        buffer = []
        for row in rows:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(row)
                continue
            i = rng.randrange(len(buffer))
            yield buffer[i]
            buffer[i] = row
        rng.shuffle(buffer)
        yield from buffer

    def _read(self, shards: list[Path]):
        for shard in shards:
            check_schema(shard)
            for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=self.read_batch_rows, columns=list(REQUIRED_COLUMNS)
            ):
                yield from zip(*(batch.column(name).to_pylist() for name in REQUIRED_COLUMNS))

    def __iter__(self):
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        for row in self.iter_rows(worker_id, num_workers):
            yield encode_row(*row)


def make_loader(dataset: ParquetPositions, batch_size: int, num_workers: int, **kwargs) -> DataLoader:
    """Build a DataLoader for a ParquetPositions dataset.

    Workers start with "forkserver", not Linux's default "fork": pyarrow and torch
    run thread pools, and forking a process with live threads can deadlock a
    worker on a lock some other thread held (Python warns about this). Don't pass
    persistent_workers=True: workers would keep their copy of the dataset, and
    set_epoch would no longer reach them.
    """
    if num_workers > 0:
        kwargs.setdefault("multiprocessing_context", "forkserver")
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)


def check_schema(shard: Path) -> None:
    """Raise ValueError if a shard is missing a required column."""
    names = set(pq.read_schema(shard).names)
    missing = [c for c in REQUIRED_COLUMNS if c not in names]
    if missing:
        raise ValueError(f"{shard} is missing columns {missing}; see chesszero/dataset.py")
