"""Tests for chesszero/encoding.py.

Input planes: layout, side-to-move flip, en passant, repetition, and that a live
board and one rebuilt from its FEN (as the supervised dataset does) encode the same.

Move index: every legal move maps to a distinct slot and back, the mask's support
is exactly the legal moves, and hand-worked indices match.
"""

import random
from collections import Counter

import chess
import numpy as np
import pytest

from chesszero.encoding import (
    INPUT_PLANES,
    POLICY_SIZE,
    board_to_planes,
    index_to_move,
    legal_move_mask,
    legal_moves_and_indices,
    move_to_index,
)


def play(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def random_games(n_games: int, max_plies: int = 200, seed: int = 0):
    """Yield (board, repetition_count) at every position of n_games random games.

    repetition_count is computed the way make_rows.py should: replay the game and
    count earlier occurrences of the position key (first four FEN fields).
    """
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board()
        seen = Counter()
        for _ in range(max_plies):
            key = " ".join(board.fen().split()[:4])
            yield board, seen[key]
            seen[key] += 1
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))


def test_shape_and_dtype():
    planes = board_to_planes(chess.Board())
    assert planes.shape == (INPUT_PLANES, 8, 8)
    assert planes.dtype == np.float32


def test_start_position():
    planes = board_to_planes(chess.Board())
    assert planes[0, 1].sum() == 8 and planes[0].sum() == 8  # own pawns on rank 2
    assert planes[6, 6].sum() == 8 and planes[6].sum() == 8  # opponent pawns on rank 7
    assert planes[5, 0, 4] == 1 and planes[5].sum() == 1  # own king on e1
    assert planes[11, 7, 4] == 1 and planes[11].sum() == 1  # opponent king on e8
    assert planes[12:14].sum() == 0
    assert np.all(planes[14] == 1)
    assert np.all(planes[15:19] == 1)
    assert planes[19].sum() == 0
    assert np.all(planes[20] == 0)


def test_black_to_move_is_flipped():
    planes = board_to_planes(play("e4"))
    # Black's own pawns appear on rank index 1, its king on [0, 4] (e8 mirrored).
    assert planes[0, 1].sum() == 8
    assert planes[5, 0, 4] == 1
    # White's e4 pawn is seen from Black's side, on e5.
    assert planes[6, 4, 4] == 1
    assert np.all(planes[14] == 0)


def test_mirrored_position_encodes_the_same():
    """A position and its colour-swapped mirror differ only in the side-to-move plane."""
    for board, _ in random_games(20, seed=1):
        a = board_to_planes(board)
        b = board_to_planes(board.mirror())
        assert np.array_equal(np.delete(a, [12, 13, 14], axis=0), np.delete(b, [12, 13, 14], axis=0))
        assert np.all(a[14] != b[14])


def test_en_passant_only_when_legal():
    # After 1.e4 the board's ep_square is e3, but no capture is possible.
    board = play("e4")
    assert board.ep_square is not None
    assert board_to_planes(board)[19].sum() == 0

    # 1.e4 a6 2.e5 d5: White can take exd6, so d6 is marked (White to move, no flip).
    planes = board_to_planes(play("e4", "a6", "e5", "d5"))
    assert planes[19, 5, 3] == 1 and planes[19].sum() == 1

    # 1.a3 e5 2.a4 e4 3.d4: Black can take exd3; d3 seen from Black's side is d6.
    planes = board_to_planes(play("a3", "e5", "a4", "e4", "d4"))
    assert planes[19, 5, 3] == 1 and planes[19].sum() == 1


def test_repetition_planes():
    shuffle = ["Nf3", "Nf6", "Ng1", "Ng8"]
    assert board_to_planes(chess.Board())[12:14].sum() == 0

    planes = board_to_planes(play(*shuffle))
    assert np.all(planes[12] == 1) and np.all(planes[13] == 0)

    planes = board_to_planes(play(*shuffle, *shuffle))
    assert np.all(planes[12] == 1) and np.all(planes[13] == 1)

    # Counts above 2 still set both planes.
    planes = board_to_planes(play(*shuffle, *shuffle, *shuffle))
    assert np.all(planes[12:14] == 1)


@pytest.mark.parametrize("count, expected", [(0, (0, 0)), (1, (1, 0)), (2, (1, 1)), (5, (1, 1))])
def test_explicit_repetition_count(count, expected):
    planes = board_to_planes(chess.Board(), repetition_count=count)
    assert (planes[12, 0, 0], planes[13, 0, 0]) == expected


def test_halfmove_clock():
    board = chess.Board("8/8/8/4k3/8/8/8/4K2R w K - 37 80")
    assert np.allclose(board_to_planes(board)[20], 0.37)


def test_fen_rebuild_matches_live_board():
    """What the supervised dataset sees (FEN + repetition column) equals what MCTS sees."""
    n = 0
    for board, repetition_count in random_games(100, seed=2):
        live = board_to_planes(board)
        rebuilt = board_to_planes(chess.Board(board.fen()), repetition_count=repetition_count)
        assert np.array_equal(live, rebuilt), board.fen()
        n += 1
    assert n > 5000


# --- Move index -------------------------------------------------------------

# Positions with promotions (straight and both captures) for both colours, castling
# both ways, and en passant, so the round trip doesn't rely on random games for them.
SPECIAL_FENS = [
    "1n1n4/2P5/8/8/8/8/k7/4K3 w - - 0 1",  # white c7: push, take b8, take d8
    "4k3/8/8/8/8/8/2p5/1N1NK3 b - - 0 1",  # black c2: push, take b1, take d1
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # white castles both ways
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",  # black castles both ways
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",  # white en passant
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2",  # black en passant
]


def all_test_boards():
    for fen in SPECIAL_FENS:
        yield chess.Board(fen)
    for board, _ in random_games(200, max_plies=300, seed=3):
        yield board


def test_round_trip_every_legal_move():
    n_moves = 0
    underpromotions = 0
    for board in all_test_boards():
        moves, indices = legal_moves_and_indices(board)
        assert len(set(indices.tolist())) == len(moves), board.fen()
        for move, index in zip(moves, indices):
            assert 0 <= index < POLICY_SIZE
            assert move_to_index(move, board) == index
            assert index_to_move(index, board) == move, (board.fen(), move.uci())
            underpromotions += move.promotion not in (None, chess.QUEEN)
        n_moves += len(moves)
    assert n_moves > 100_000
    assert underpromotions > 0


def test_mask_support_is_legal_moves():
    for board in all_test_boards():
        mask = legal_move_mask(board)
        assert mask.shape == (POLICY_SIZE,) and mask.dtype == np.bool_
        expected = {move_to_index(m, board) for m in board.legal_moves}
        assert set(np.flatnonzero(mask).tolist()) == expected


def test_worked_examples():
    board = chess.Board()
    assert move_to_index(chess.Move.from_uci("e2e4"), board) == 76  # N, distance 2: plane 1
    assert move_to_index(chess.Move.from_uci("g1f3"), board) == 4038  # knight plane 63

    # Black's e7e5 is White's e2e4 seen from Black's side: the same slot.
    board.push_uci("e2e4")
    assert move_to_index(chess.Move.from_uci("e7e5"), board) == 76

    promo = chess.Board(SPECIAL_FENS[0])
    assert move_to_index(chess.Move.from_uci("c7c8n"), promo) == (64 + 1 * 3 + 0) * 64 + 50
    assert move_to_index(chess.Move.from_uci("c7b8r"), promo) == (64 + 0 * 3 + 2) * 64 + 50
    assert move_to_index(chess.Move.from_uci("c7d8b"), promo) == (64 + 2 * 3 + 1) * 64 + 50
    # Queen promotion shares the queen-style plane (N, distance 1 = plane 0).
    assert move_to_index(chess.Move.from_uci("c7c8q"), promo) == 0 * 64 + 50
    assert index_to_move(50, promo) == chess.Move.from_uci("c7c8q")

    # Castling is the king's two-square move: E distance 2 = plane 15, from e1.
    castle = chess.Board(SPECIAL_FENS[2])
    assert move_to_index(chess.Move.from_uci("e1g1"), castle) == 15 * 64 + 4
    assert index_to_move(15 * 64 + 4, castle) == chess.Move.from_uci("e1g1")


def test_index_to_move_rejects_masked_indices():
    board = chess.Board()
    illegal = np.flatnonzero(~legal_move_mask(board))
    rng = np.random.default_rng(0)
    for index in rng.choice(illegal, size=200, replace=False):
        with pytest.raises(ValueError):
            index_to_move(index, board)
