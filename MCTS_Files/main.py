import chess
from mcts import MCTS

board = chess.Board()

mcts = MCTS(
    exploration_constant=2,
    stockfish_nodes=1000
)

try:
    best_move = mcts.search(board, iterations=100)

    print("Selected move:", best_move)

    if best_move is not None:
        board.push(best_move)
        print(board)

finally:
    mcts.evaluator.close()