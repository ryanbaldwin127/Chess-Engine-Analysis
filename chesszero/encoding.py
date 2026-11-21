"""Board to planes, move <-> index, legal-move mask.

This module is the contract between the board and the network. The model, the
loss, the dataset and the MCTS all depend on these shapes and index layouts, so
change them only together with chesszero/model.py and the tests.

Everything is encoded from the side to move's view. When Black is to move, the
board is mirrored vertically so that the side to move's pieces always start at
the bottom, and "own" / "opponent" replace White / Black.

Input planes (INPUT_PLANES = 21), each 8x8, indexed [plane, rank, file] with
rank 0 being the side to move's back rank:

    0-5    own pawns, knights, bishops, rooks, queens, king (1 where present)
    6-11   opponent pieces, same order
    12     current position has occurred once before (all 1s or all 0s)
    13     current position has occurred twice before (all 1s or all 0s)
    14     side to move is White (all 1s or all 0s)
    15-16  own kingside / queenside castling right (all 1s or all 0s)
    17-18  opponent kingside / queenside castling right (all 1s or all 0s)
    19     en passant target square (a single 1, else all 0s)
    20     halfmove clock / 100 (same value on every square)

Policy index layout (POLICY_SIZE = 4672 = 73 move planes x 64 squares), as in
AlphaZero [AlphaZero]:

    index = move_plane * 64 + from_square

    from_square is the move's origin square from the side to move's view
    (python-chess numbering, a1 = 0 ... h8 = 63, mirrored when Black is to move).

    move_plane 0-55   "queen-style" moves: 8 directions x distance 1-7.
                      Direction order: N, NE, E, SE, S, SW, W, NW;
                      plane = direction * 7 + (distance - 1).
                      Promotions to a queen use these planes.
    move_plane 56-63  knight moves, in the order of KNIGHT_OFFSETS below.
    move_plane 64-72  underpromotions: 3 directions (capture left, straight,
                      capture right) x 3 pieces (knight, bishop, rook);
                      plane = 64 + direction * 3 + piece.

This "plane-major" order matches the network's convolutional policy head, which
outputs a (73, 8, 8) tensor that is flattened to 4672 logits.
"""

import chess
import numpy as np

INPUT_PLANES = 21
NUM_MOVE_PLANES = 73
POLICY_SIZE = NUM_MOVE_PLANES * 64  # 4672

# (file_delta, rank_delta) for each knight move plane, from the side to move's view.
KNIGHT_OFFSETS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a position as a float32 array of shape (INPUT_PLANES, 8, 8).

    See the module docstring for the plane layout. Repetition planes come from
    board.is_repetition(2) and board.is_repetition(3), so the board must keep its
    move stack (don't pass copies made with stack=False).
    """
    raise NotImplementedError("Ryan: board_to_planes")


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Return the policy index in [0, POLICY_SIZE) for a legal move in this position."""
    raise NotImplementedError("move_to_index")


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of move_to_index for this position. Returns a legal chess.Move.

    Promotions to a queen come from the queen-style planes, so a pawn reaching the
    last rank on those planes must be returned with promotion=chess.QUEEN.
    """
    raise NotImplementedError("index_to_move")


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a bool array of shape (POLICY_SIZE,), True exactly at the legal moves' indices."""
    raise NotImplementedError("legal_move_mask")
