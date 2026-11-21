import chess
from chess_mcts import MCTS

board = chess.Board()

mcts = MCTS(
    exploration_constant=2,
    stockfish_nodes=1000
)

try:
    print("Starting position:")
    print(board)

    print("\nStarting MCTS search...")
    best_move = mcts.search(board, iterations=100)

    print("Selected move:", best_move)
    for child in mcts.root.children:
        print(
            f"Move: {child.action}, "
            f"Visits: {child.visits}, "
            f"Reward: {1.0 - child.mean_reward():.3f}"
        )

    if best_move is not None:
        board.push(best_move)
        print("\nBoard after MCTS Move")
        print(board)

finally:
    mcts.evaluator.close()