"""Test for chesszero/players.py and chesszero/game.py
Run from the repo root: python -m pytest tests/test_game.py -v
Stockfish tests auto skipped if no Stockfish binary found
"""

import chess
import chess.pgn
import pytest

from chesszero.game import play_game, save_pgn
from chesszero.players import Player, RandomPlayer, StockfishPlayer, find_stockfish


class ScriptedPlayer(Player):

    def __init__(self, moves, name="Scripted"):
        self.moves = iter(moves)
        self.name = name

    def choose_move(self, board):
        return chess.Move.from_uci(next(self.moves))


def scripted_pair(uci_moves):
    return ScriptedPlayer(uci_moves[0::2], "W"), ScriptedPlayer(uci_moves[1::2], "B")


def has_stockfish():
    try:
        find_stockfish()
        return True
    except FileNotFoundError:
        return False


# -------------- Game runner --------------

def test_games_end():
    valid_results = {"1-0", "0-1", "1/2-1/2"}
    for i in range(200):
        r = play_game(RandomPlayer(seed=i), RandomPlayer(seed=10_000 + i))
        assert r.result in valid_results
        assert 0 < r.plies <= 300


def test_seeds_repeat(): 
    a = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2))
    b = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2))
    assert a.final_fen == b.final_fen
    assert a.plies == b.plies


def test_checkmate():
    w, b = scripted_pair(["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"])
    r = play_game(w, b)
    assert r.result == "1-0"
    assert r.termination == "checkmate"
    assert r.plies == 7


def test_repetition_draw():
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"] * 3
    w, b = scripted_pair(shuffle)
    r = play_game(w, b)
    assert r.result == "1/2-1/2"
    assert r.termination == "threefold_repetition"
    assert r.plies <= 8


def test_move_cap():
    r = play_game(RandomPlayer(seed=0), RandomPlayer(seed=1), max_plies=10)
    assert r.plies == 10
    assert r.result == "1/2-1/2"
    assert r.termination == "move_cap"


def test_illegal_move():
    w, b = scripted_pair(["e2e5"])
    with pytest.raises(ValueError):
        play_game(w,b)


def test_start_position():
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    w, b = scripted_pair(["a1a8"])
    r = play_game(w, b, start_fen=fen)
    assert r.result == "1-0"
    assert r.pgn.headers["FEN"] == fen


def test_save_pgn(tmp_path):
    path = tmp_path / "games.pgn"
    results = [play_game(RandomPlayer(seed=i), RandomPlayer(seed=i + 1)) for i in range(3)]
    for r in results:
        save_pgn(r.pgn, path)

    with open(path, encoding="utf-8") as f:
        for r in results:
            game = chess.pgn.read_game(f)
            assert game.headers["Result"] == r.result
            assert game.headers["Termination"] == r.termination
            assert game.end().board().fen() == r.final_fen

# -------------- Stockfish --------------

@pytest.mark.skipif(not has_stockfish(), reason="Stockfish not found")
def test_stockfish_mate():
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    with StockfishPlayer(nodes=1000) as sf:
        assert sf.choose_move(chess.Board(fen)) == chess.Move.from_uci("a1a8")

@pytest.mark.skipif(not has_stockfish(), reason="Stockfish not found")
def test_stockfish_wins():
    with StockfishPlayer(nodes=1000) as sf:
        for i in range(2):
            r = play_game(sf, RandomPlayer(seed=i))
            assert r.result == "1-0"
            r = play_game(RandomPlayer(seed=i), sf)
            assert r.result == "0-1"
