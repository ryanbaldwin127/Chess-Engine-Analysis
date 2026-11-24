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
    12     current position has occurred at least once before (all 1s or all 0s)
    13     current position has occurred at least twice before (all 1s or all 0s)
    14     side to move is White (all 1s or all 0s)
    15-16  own kingside / queenside castling right (all 1s or all 0s)
    17-18  opponent kingside / queenside castling right (all 1s or all 0s)
    19     en passant target square, only if an en passant capture is legal
           (a single 1, else all 0s)
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
    move_plane 64-72  underpromotions: 3 directions (capture toward the a-file,
                      straight, capture toward the h-file) x 3 pieces (knight,
                      bishop, rook); plane = 64 + direction * 3 + piece.

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


PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _bitboards_to_planes(bitboards: list[int]) -> np.ndarray:
    """Turn 64-bit square sets into (n, 8, 8) uint8 planes, [rank, file] with a1 at [0, 0]."""
    as_bytes = np.array(bitboards, dtype="<u8").view(np.uint8).reshape(len(bitboards), 8)
    return np.unpackbits(as_bytes, axis=1, bitorder="little").reshape(len(bitboards), 8, 8)


def board_to_planes(board: chess.Board, repetition_count: int | None = None) -> np.ndarray:
    """Encode a position as a float32 array of shape (INPUT_PLANES, 8, 8).

    See the module docstring for the plane layout.

    repetition_count is how many times this position occurred earlier in the game
    (0, 1, 2 or more). Leave it as None for a live board with its move stack: it is
    then read from board.is_repetition, so don't pass copies made with stack=False.
    Pass it explicitly for a board rebuilt from a FEN, which has no history; the
    supervised rows carry it as a column (docs/decisions.md #13).

    The en passant plane is set only when an en passant capture is legal, matching
    what board.fen() writes, so a live board and one rebuilt from its FEN encode
    the same (board.ep_square is set after every double pawn push).
    """
    us = board.turn
    them = not us

    bitboards = [board.pieces_mask(pt, us) for pt in PIECE_TYPES]
    bitboards += [board.pieces_mask(pt, them) for pt in PIECE_TYPES]
    if board.has_legal_en_passant():
        bitboards.append(chess.BB_SQUARES[board.ep_square])
    else:
        bitboards.append(chess.BB_EMPTY)
    if us == chess.BLACK:
        bitboards = [chess.flip_vertical(bb) for bb in bitboards]

    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    square_planes = _bitboards_to_planes(bitboards)
    planes[0:12] = square_planes[:12]
    planes[19] = square_planes[12]

    if repetition_count is None:
        # is_repetition(n): the position occurred at least n times, counting now.
        repetition_count = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
    planes[12] = repetition_count >= 1
    planes[13] = repetition_count >= 2
    planes[14] = us == chess.WHITE
    planes[15] = board.has_kingside_castling_rights(us)
    planes[16] = board.has_queenside_castling_rights(us)
    planes[17] = board.has_kingside_castling_rights(them)
    planes[18] = board.has_queenside_castling_rights(them)
    planes[20] = board.halfmove_clock / 100.0
    return planes


# (file_delta, rank_delta) unit steps for the queen-style directions, in plane order.
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

# Underpromotion pieces in plane order; directions are file_delta -1, 0, +1 (toward
# the a-file, straight, toward the h-file; files are not mirrored for Black).
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _build_tables() -> tuple[np.ndarray, np.ndarray]:
    """Precompute the move-plane geometry, all from the side to move's view.

    plane_of[from, to]: the queen-style or knight plane for a move, or -1.
    target_of[plane, from]: the destination square of that plane's move, or -1 if
    it leaves the board. Underpromotion planes are one rank forward.
    """
    plane_of = np.full((64, 64), -1, dtype=np.int64)
    target_of = np.full((NUM_MOVE_PLANES, 64), -1, dtype=np.int64)

    def on_board(file, rank):
        return 0 <= file < 8 and 0 <= rank < 8

    for sq in range(64):
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        for d, (df, dr) in enumerate(QUEEN_DIRECTIONS):
            for dist in range(1, 8):
                f, r = file + df * dist, rank + dr * dist
                if on_board(f, r):
                    plane = d * 7 + (dist - 1)
                    plane_of[sq, chess.square(f, r)] = plane
                    target_of[plane, sq] = chess.square(f, r)
        for k, (df, dr) in enumerate(KNIGHT_OFFSETS):
            f, r = file + df, rank + dr
            if on_board(f, r):
                plane_of[sq, chess.square(f, r)] = 56 + k
                target_of[56 + k, sq] = chess.square(f, r)
        for d, df in enumerate((-1, 0, 1)):
            f, r = file + df, rank + 1
            if on_board(f, r):
                for p in range(3):
                    target_of[64 + d * 3 + p, sq] = chess.square(f, r)
    return plane_of, target_of


_PLANE_OF, _TARGET_OF = _build_tables()


def _relative(square: int, turn: chess.Color) -> int:
    """Map a square to or from the side to move's view (a vertical mirror for Black)."""
    return square if turn == chess.WHITE else square ^ 56


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Return the policy index in [0, POLICY_SIZE) for a legal move in this position.

    Only the move's geometry and the side to move are used, so this doesn't check
    legality; an impossible shape (e.g. a1 to b4) raises ValueError. Castling is
    the king's two-square move (e1g1), as python-chess gives it for standard chess.
    """
    from_sq = _relative(move.from_square, board.turn)
    to_sq = _relative(move.to_square, board.turn)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        df = chess.square_file(to_sq) - chess.square_file(from_sq)
        plane = 64 + (df + 1) * 3 + UNDERPROMOTIONS.index(move.promotion)
    else:
        plane = int(_PLANE_OF[from_sq, to_sq])
        if plane < 0:
            raise ValueError(f"{move.uci()} has no move plane")
    return plane * 64 + from_sq


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of move_to_index for this position. Returns a legal chess.Move.

    Promotions to a queen come from the queen-style planes, so a pawn reaching the
    last rank on those planes is returned with promotion=chess.QUEEN. Raises
    ValueError if the index isn't a legal move here (i.e. it was masked).
    """
    plane, from_sq = divmod(int(index), 64)
    to_sq = int(_TARGET_OF[plane, from_sq]) if 0 <= plane < NUM_MOVE_PLANES else -1
    if to_sq < 0:
        raise ValueError(f"index {index} leaves the board")

    promotion = None
    if plane >= 64:
        promotion = UNDERPROMOTIONS[(plane - 64) % 3]
    elif chess.square_rank(to_sq) == 7 and board.piece_type_at(_relative(from_sq, board.turn)) == chess.PAWN:
        promotion = chess.QUEEN

    move = chess.Move(_relative(from_sq, board.turn), _relative(to_sq, board.turn), promotion)
    if not board.is_legal(move):
        raise ValueError(f"index {index} ({move.uci()}) is not legal in {board.fen()}")
    return move


def legal_moves_and_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    """Return the legal moves and their policy indices, in the same order.

    MCTS can use this to expand a node once: gather each child's prior from the
    policy logits at these indices, instead of building the mask and then
    generating the moves again.
    """
    moves = list(board.legal_moves)
    indices = np.fromiter((move_to_index(m, board) for m in moves), dtype=np.int64, count=len(moves))
    return moves, indices


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a bool array of shape (POLICY_SIZE,), True exactly at the legal moves' indices."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    mask[legal_moves_and_indices(board)[1]] = True
    return mask
