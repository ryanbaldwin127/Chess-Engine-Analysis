import chess
from chess_mcts import MCTS

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.mcts = MCTS()

    def selectionMethodTest(self):
        root = self.mcts.create_a_node(self.board)
        selected = self.mcts.selection(root)

        print("TEST 1: Unexpanded Root Node")
        print("Selected Node: ", selected.node_id)
        print("Expected Node: ", root.node_id)
        print("Passed: ", selected == root, "\n")

        while not root.is_fully_expanded():
            self.mcts.expansion(root)
        
        root.visits = 10
        i = 1
        for child in root.children:
            child.visits = 2
            child.total_reward = 1.0 + i
            i += 3
        
        selected = self.mcts.selection(root)
        
        print("TEST 2: Fully Expanded Root Node - Exploitation")
        print("Selected Node: ", selected.node_id)
        print("Expected Node: ", selected in root.children)
        print("Updated Total Reward: ", selected.total_reward)
        print("Passed: ", selected in root.children, "\n")

        for i, child in enumerate(root.children):
            child.visits = 2 + i
            child.total_reward = child.visits * 0.5
        root.visits = sum(child.visits for child in root.children)
        selected = self.mcts.selection(root)

        print("TEST 3: Fully Expanded Root Node - Exploration")
        print("Selected Node: ", selected.node_id)
        print("Expected Node: ", selected in root.children)
        print("Updated Total Reward: ", selected.total_reward)
        print("Selected Mean Reward: ", selected.mean_reward())
        print("Child Visits: ", selected.visits)
        print("Passed: ", selected in root.children, "\n")

    def expansionMethodTest(self):
        root = self.mcts.create_a_node(self.board)
        child = self.mcts.expansion(root)

        print("Root Children: ", len(root.children))
        print("Root Unexplored Moves: ", len(root.untried_moves))
        print("Child Move: ", child.action)
        print("Child Player: ", child.player)
        print("Child Legal Moves: ", len(child.untried_moves))

if __name__ == "__main__":
    test = Game()
    test.selectionMethodTest()
    test.expansionMethodTest()