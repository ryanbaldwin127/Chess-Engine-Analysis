import math
import random
import chess
from node import Node
from stockfishEvaluator import StockfishEvaluator

class MCTS:
    def __init__(self, exploration_constant = 2, stockfish_nodes = 1000):
        self.root = None
        self.exploration_constant = exploration_constant
        self.node_count = 0

        self.evaluator = StockfishEvaluator(
            nodes = stockfish_nodes
        )
    
    '''
    create_a_node method takes a given board state and creates the root based on that to be expanded.
    It uses the Node class to get a copy of the board state, initializes the player to be whoevers turn
    and uses it's own count as it's unique identifier, then the node count goes up by 1 because there is
    now and existing node in the tree.  The number of untried moves is initialized to the collection of
    all legal moves that can be made.
    '''

    def create_a_node(self, board):
        self.root = Node(
            state = board.copy(),
            player = board.turn,
            node_id = self.node_count
        )

        self.node_count += 1
        self.root.untried_moves = list(board.legal_moves)

        return self.root

    '''
    The selection method loops while it has a node that isn't in a terminal state and checks if the node 
    hasn't been fully expanded.  If the node is fully expanded, it takes the node with the highest
    uct score and does this across multiple partitions.
    '''

    def selection(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            if not node.children:
                return node
            
            node = max(node.children,
                        key = lambda child: child.uct_score(node, self.exploration_constant))
        return node

    '''
    Below is the expansion method.  First, we check to see if the node in question is fully expanded (seeing if there are
    any unexplored moves).  Then, we select and remove an unexplored move from the nodes list and copy the current chess
    position.  We apply the selected move on the new state of the game and determine who the next player is.  We create a new
    node with the updated parameters and state, reinitialize the child's unexplored legal moves and attach it to the parent 
    which was the original node.
    '''    

    def expansion(self, node):
        if node.is_terminal or node.is_fully_expanded():
            return node

        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        new_state = node.state.copy()
        new_state.push(move)
        next_player = new_state.turn

        child = Node(
            state = new_state,
            player = next_player,
            parent = node,
            action = move,
            node_id = self.node_count
        )

        self.node_count += 1
        child.untried_moves = list(new_state.legal_moves)
        node.children.append(child)

        return child

    def simulation(self, node):
        board = node.state.copy()
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)

        result = board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else: 
            return 0
    
    def stockfish_simulation(self, node):
        board = node.state.copy()

        if board.is_checkmate():
            return 0.0
        if board.is_game_over():
            return 0.5
        score = self.evaluator.evaluate(board)
        reward = 1.0 / (1.0 + math.exp(-score/2.0))
        
        return reward
    
    def backpropogation(self, node, result):
        while node is not None:
            node.visits += 1
            node.total_reward += result
            
            #The result here is to switch the perspective for previous player
            result = 1.0 - result
            node = node.parent

    #Here we are running the full Monte Carlo Tree Search algorithm within this file's main:
    def search(self, board, iterations):
        self.create_a_node(board)
        for _ in range(iterations):
            #Selection method
            node = self.selection(self.root)
            #Expansion method
            node = self.expansion(node)
            #Simulate Moves to a terminal state
            #result = self.simulation(node)
            result = self.stockfish_simulation(node)
            # Propogate result back through the tree and update nodes along the way
            self.backpropogation(node, result)

        if not self.root.children:
            return None
        
        best_child = max(
            self.root.children, 
            key = lambda child: child.visits
        )
        return best_child.action

    




