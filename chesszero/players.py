"""Player interface plus Random, Stockfish, and MCTS players.
Every player shares one method choose_move(board) -> chess.Move, so game runner and
eval code can use any of them the same way.
    
    Example:
        with StockfishPlayer(nodes=1000) as sf:
            result = play_game(sf, RandomPlayer(seed=0))
"""

import os
import random
import shutil

import chess
import chess.engine


SHARED_STOCKFISH = "/s/bach/a/class/cs435/cs435a/chess-zero/engines/stockfish-19/stockfish"


def find_stockfish():
    candidates = [os.environ.get("STOCKFISH_PATH"), SHARED_STOCKFISH, shutil.which("stockfish")]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "Could not find StockFish. Set STOCKFISH_PATH, or put the binary at " + SHARED_STOCKFISH)

class Player:
    
    
    name = "Player"

    def choose_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return self.name


class RandomPlayer(Player):


    def __init__(self, seed=None, name="Random"):
        self.rng = random.Random(seed)
        self.name = name

    def choose_move(self, board):
        return self.rng.choice(list(board.legal_moves))


class StockfishPlayer(Player):


    def __init__(self, path=None, nodes=None, depth=None, skill_level=None,
                threads=1, hash_mb=16, name=None):
        if nodes is None and depth is None:
            nodes = 10_000
        self.limit = chess.engine.Limit(nodes=nodes, depth=depth)
        self.engine = chess.engine.SimpleEngine.popen_uci(path or find_stockfish())

        options = {"Threads": threads, "Hash": hash_mb}
        if skill_level is not None:
            options["Skill Level"] = skill_level
        self.engine.configure(options)

        if name is None:
            parts = ["Stockfish"]
            if skill_level is not None:
                parts.append(f"skill{skill_level}")
            parts.append(f"nodes{nodes}" if nodes is not None else f"depth{depth}")
            name = "-".join(parts)
        self.name = name

    def choose_move(self, board):
        result = self.engine.play(board, self.limit)
        if result.move is None:
            raise RuntimeError(f"{self.name} returned no move for {board.fen()}")
        return result.move

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None


class MCTSPlayer(Player):
    """Placeholder until mcts.py and model.py are written.
    This should load a checkpoint, run the search on the board, and return the chosen move.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MCTSPlayer needs mcts.py and model.py first"
        )