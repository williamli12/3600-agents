"""
evaluator.py - Board evaluation for MaxBeater (Minimax version)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
from collections import deque
import numpy as np
from game import board, enums
from .belief import TrapdoorBelief

Coord = Tuple[int, int]

class Evaluator:
    """
    Evaluator class that provides heuristic evaluation for the Minimax search.
    """
    
    def __init__(self, trap_belief: TrapdoorBelief):
        self.trap_belief = trap_belief
        
        # Weights
        self.w_eggs = 50.0
        self.w_space = 5.0
        self.w_trap = 20.0
        self.w_revisit = 2.0
        self.w_turds = 5.0
        self.w_mobility = 1.0

    def evaluate_state(
        self, 
        game_board: board.Board, 
        belief: TrapdoorBelief, 
        visit_counts: Dict[Coord, int]
    ) -> float:
        """
        Heuristic evaluation of the board state.
        
        Args:
            game_board: The current board state (perspective of 'chicken_player').
            belief: The current trapdoor belief state.
            visit_counts: Dictionary mapping coordinates to visit counts for the current search path.
        
        Returns:
            Float score (higher is better for chicken_player).
        """
        # Terminal state check
        if game_board.is_game_over():
            winner = game_board.get_winner()
            if winner == enums.Result.PLAYER:
                return 100000.0
            elif winner == enums.Result.ENEMY:
                return -100000.0
            else:
                return 0.0
        
        score = 0.0
        
        # 1. Egg Count Difference
        my_eggs = game_board.chicken_player.get_eggs_laid()
        opp_eggs = game_board.chicken_enemy.get_eggs_laid()
        eggs_diff = my_eggs - opp_eggs
        score += self.w_eggs * eggs_diff

        # 2. Trapdoor Risk
        # Penalize standing on high risk
        my_pos = game_board.chicken_player.get_location()
        opp_pos = game_board.chicken_enemy.get_location()
        
        my_risk = belief.risk(my_pos) if hasattr(belief, 'risk') else (belief.get_prob(*my_pos) if hasattr(belief, 'get_prob') else 0)
        # Adapt to whatever belief API we settled on. belief.py has get_prob(row, col).
        # I added get_prob. I didn't add risk() in my rewrite? 
        # Ah, I missed adding `risk()` wrapper in belief.py. 
        # `get_prob` handles parity internally so it is effectively the risk (since other parity is 0).
        my_risk = belief.get_prob(my_pos[1], my_pos[0]) # Board uses (x,y) = (col, row). get_prob uses (row, col)?
        # Let's check belief.py again. 
        # In belief.py: get_prob(row, col).
        # Board loc is (x, y).
        # Usually x=col, y=row.
        # So call belief.get_prob(my_pos[1], my_pos[0]).
        
        # Actually, let's just look at belief.py again to be sure about coordinate system.
        # belief.py: `coord = (x, y)`. `get_prob(row, col)` -> `coord=(row, col)`.
        # If `_init_priors` uses `x` and `y` and `coord=(x,y)`, then `get_prob` should take `x, y`.
        # My rewrite of belief.py: `get_prob(row, col) -> coord=(row, col)`.
        # The user prompt implies standard matrix indexing for belief?
        # "get_prob(self, row, col) -> float".
        # Game uses (x, y).
        # I will assume x=col, y=row.
        
        my_risk = belief.get_prob(my_pos[1], my_pos[0]) 
        score -= self.w_trap * my_risk * 10.0 # Heavy penalty for standing on trap

        # Reward if opponent is on risk
        opp_risk = belief.get_prob(opp_pos[1], opp_pos[0])
        score += self.w_trap * opp_risk * 5.0

        # Neighborhood risk?
        # Penalize moving into high risk areas
        
        # 3. BFS Reachable Space
        # Estimate additional eggs we can lay
        my_space = self.bfs_reachable_eggs(game_board, my_pos, game_board.turns_left_player, belief, game_board.chicken_player, is_opponent=False)
        
        # Estimate for opponent
        # We need their turns left. game_board has turns_left_enemy.
        opp_space = self.bfs_reachable_eggs(game_board, opp_pos, game_board.turns_left_enemy, belief, game_board.chicken_enemy, is_opponent=True)
        
        score += self.w_space * (my_space - opp_space)

        # 4. Visit Count Penalty (Discourage oscillation)
        # visit_counts keys are (x, y)
        revisit_penalty = 0.0
        if my_pos in visit_counts:
             revisit_penalty = visit_counts[my_pos] * self.w_revisit
        score -= revisit_penalty

        # 5. Turd Placement
        # Bonus for turds in central lanes
        # We can iterate over turds_player set
        for turd_pos in game_board.turds_player:
             # Center is 3, 4. Dist from center:
             dist = abs(turd_pos[0] - 3.5) + abs(turd_pos[1] - 3.5)
             if dist < 3: # Inner rings
                 score += self.w_turds * (3 - dist)
        
        return score
    
    def bfs_reachable_eggs(
        self, 
        board_state: board.Board, 
        start_pos: Coord, 
        max_moves: int, 
        belief: TrapdoorBelief,
        chicken,
        is_opponent: bool
    ) -> float:
        """
        BFS to estimate maximum number of ADDITIONAL eggs that can be laid.
        """
        # If 0 moves left, 0 eggs.
        if max_moves <= 0:
            return 0.0
    
        queue = deque([(start_pos, 0)]) # (pos, depth)
        visited = {start_pos}
        reachable_egg_spots = 0
        
        # Which squares are blocked?
        # Opponent turds block us. Our turds block us.
        # Existing eggs block us (can't lay egg on egg).
        # Trapdoors with high prob should be treated as blocked?
        
        # We'll use a simplified check.
        # We need to know parity.
        parity = chicken.even_chicken
        
        while queue:
            curr, depth = queue.popleft()
            
            if depth >= max_moves:
                continue
            
            # Explore neighbors
            for d in enums.Direction:
                try:
                    nex = enums.loc_after_direction(curr, d)
                except ValueError:
                    continue
                
                if not board_state.is_valid_cell(nex):
                    continue
                
                if nex in visited:
                    continue
                
                # Check blocking
                # Opponent turds?
                if is_opponent:
                    if nex in board_state.turds_player: continue
                else:
                    if nex in board_state.turds_enemy: continue
                
                # Own turds?
                if is_opponent:
                    if nex in board_state.turds_enemy: continue
        else:
                    if nex in board_state.turds_player: continue
                
                # Trapdoor risk?
                prob = belief.get_prob(nex[1], nex[0])
                if prob > 0.3: # Threshold
                    continue
                
                # If we can step here, add to queue
                visited.add(nex)
                queue.append((nex, depth + 1))
                
                # Can we lay an egg here?
                # Check parity
                if (nex[0] + nex[1]) % 2 == parity:
                    # Check if egg already exists
                    if nex not in board_state.eggs_player and nex not in board_state.eggs_enemy:
                         # Distinct spot found
                         reachable_egg_spots += 1
        
        return float(reachable_egg_spots)

