"""Play two players, draw rules, move cap, save PGN.
Games end on checkmate, stalemate, or a claimable draw (repetition, fifty-move rule). 
Games that reach max_plies are scored as draws.
Illegal moves raise errors.

Example:
    result = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2))
    save_pgn(result.pgn, "games.pgn")
"""

import datetime
from dataclasses import dataclass

import chess
import chess.pgn

DEFAULT_MAX_PLIES = 300


@dataclass
class GameResult:
    result: str
    termination: str
    plies: int
    pgn: chess.pgn.Game
    final_fen: str


def play_game(white, black, start_fen=None, max_plies=DEFAULT_MAX_PLIES, 
              event="CS435 game", round_label="-"):
    board = chess.Board(start_fen) if start_fen else chess.Board()
    players = {chess.WHITE: white, chess.BLACK: black}
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        player = players[board.turn]
        move = player.choose_move(board.copy())

        if move not in board.legal_moves:
            raise ValueError(
                f"{player!r} played illegal move {move} in position {board.fen()}")
        
        board.push(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result = outcome.result()
        termination = outcome.termination.name.lower()
    else:
        result = "1/2-1/2"
        termination = "move_cap"

    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = event
    game.headers["Site"] = "CS435"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["Round"] = str(round_label)
    game.headers["White"] = getattr(white, "name", type(white).__name__)
    game.headers["Black"] = getattr(black, "name", type(black).__name__)
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    game.headers["PlyCount"] = str(plies)

    return GameResult(result=result, termination=termination, plies=plies, pgn=game, final_fen = board.fen())


def save_pgn(game, path, append=True):
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        print(game, file=f, end="\n\n")