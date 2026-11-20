import chess
import chess.engine

class StockfishEvaluator:
    def __init__(self, nodes = 1000):
        engine = chess.engine.SimpleEngine.popen_uci("/s/bach/a/class/cs435/cs435a/chess-zero/engines/stockfish")
        self.engine.configure({Threads: 1})
        self.nodes = nodes

    def evaluate(self, node):
        if board.is_checkmate():
            return -1000.0

        if board.is_game_over():
            return 0.0

        info = self.engine.analyse(
            board,
            chess.engine.Limit(nodes=self.nodes)
        )

        score = info["score"].pov(board.turn)

        return score.score(mate_score=100000) / 100.0

    def close(self):
        self.engine.quit()