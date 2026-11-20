import math
from typing import Any, Optional

class Node:
    def __init__(self, 
        state: Any, 
        player: int,
        parent: Optional["Node"] = None,
        action: Any = None,
        node_id: int = 0  
    ):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.node_id = node_id

        self.children = []
        self.untried_moves = []

        self.visits = 0
        self.total_reward = 0.0
        
    
    def mean_reward(self):
        if (self.visits == 0):
            return 0.0
        return self.total_reward / self.visits
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.state.is_game_over()

    def uct_score(self, parent, c = 2):
        if (self.visits == 0):
            return float("inf")
        
        if (parent.visits == 0):
            return 0.0

        exploitation = self.mean_reward()
        exploration = c * math.sqrt(math.log(parent.visits) / self.visits)

        return exploitation + exploration
    
    
    
    

        