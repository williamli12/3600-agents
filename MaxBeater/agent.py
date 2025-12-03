"""
agent.py - Main agent orchestrator for MaxBeater

MaxBeater is a strong AI agent combining:
- Bayesian trapdoor belief tracking
- Hand-crafted heuristics
- Minimax search with Alpha-Beta pruning and Transposition Table
"""

from __future__ import annotations
from collections.abc import Callable
from typing import List, Tuple, Optional
import numpy as np
from game import board, enums

# Import our components
from .belief import TrapdoorBelief
from .evaluator import Evaluator
from .search_minimax import MinimaxSearch


class PlayerAgent:
    """
    MaxBeater - Strong agent with Minimax + Heuristics.
    """
    
    def __init__(self, game_board: board.Board, time_left: Callable):
        """
        Initialize the agent and all components.
        """
        seed = 42
        self.rng = np.random.default_rng(seed)
        
        print("[MaxBeater] Initializing agent...")
        
        # === Component 1: Trapdoor Belief Tracker ===
        self.trap_belief = TrapdoorBelief(board_size=8)
        print("[MaxBeater] Trapdoor belief system initialized")
        
        # === Component 2: Evaluator ===
        self.evaluator = Evaluator(self.trap_belief)
        print("[MaxBeater] Evaluator initialized")
        
        # === Component 3: Search Engine ===
        self.minimax = MinimaxSearch(self.evaluator)
        print("[MaxBeater] Minimax search engine initialized")
        
        self.opening_phase_limit = 5
    
    def play(
        self,
        game_board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Choose and return the best move for the current turn.
        """
        # === Step 1: Update trapdoor beliefs ===
        self.trap_belief.update(game_board, sensor_data)
        
        # === Step 2: Check opening book ===
        # Only if we haven't played many turns (game_board.turn_count counts total turns, 
        # we want our turns. game_board.turn_count starts at 0, increments by 1 each turn.
        # So turn_count // 2 is roughly our move number.)
        my_move_num = game_board.turn_count // 2
        if my_move_num < self.opening_phase_limit:
            opening_move = self.choose_opening_move(game_board)
            if opening_move:
                 print(f"[MaxBeater] Using opening move: {opening_move}")
                 return opening_move

        # === Step 3: Run Minimax Search ===
        # Initialize visit counts for this search
        # We start with empty counts or current pos?
        # "carry along a small 8x8 visit_counts array... for current player's path"
        # At root, path includes current position.
        current_pos = game_board.chicken_player.get_location()
        visit_counts = {current_pos: 1}
        
        move = self.minimax.choose_best_move(game_board, time_left, self.trap_belief, visit_counts)
        
        print(f"[MaxBeater] Selected move: {move}")
        return move

    def choose_opening_move(self, game_board: board.Board) -> Optional[Tuple[enums.Direction, enums.MoveType]]:
        """
        Hard-coded opening strategy:
        Rush toward the second ring or center to claim space, avoiding edges.
        """
        my_loc = game_board.chicken_player.get_location()
        
        # If we can lay an egg, do it (unless it's bad?)
        # Actually opening usually just moves to position.
        # "rush toward the second ring or center to claim space instead of corner"
        
        # Target: (3,3), (3,4), (4,3), (4,4) or slightly wider (2..5)
        target_center = (3.5, 3.5)
        
        valid_moves = game_board.get_valid_moves(enemy=False)
        if not valid_moves: return None
        
        # 1. If we can lay an egg safely, do it?
        # The prompt says "rush to center... laying eggs along the way".
        # Check if any valid move is EGG
        for d, mtype in valid_moves:
            if mtype == enums.MoveType.EGG:
                return (d, mtype)
        
        # 2. Move towards center
        best_move = None
        min_dist = float('inf')
        
        # Filter out TURD moves in opening
        plain_moves = [(d, m) for d, m in valid_moves if m == enums.MoveType.PLAIN]
        
        if not plain_moves:
            return valid_moves[0] # Fallback
            
        for d, mtype in plain_moves:
            # Get new loc
            try:
                new_loc = enums.loc_after_direction(my_loc, d)
            except ValueError:
                        continue
                    
            # Avoid high trapdoor prob
            risk = self.trap_belief.get_prob(new_loc[1], new_loc[0])
            if risk > 0.2:
                continue
            
            dist = abs(new_loc[0] - target_center[0]) + abs(new_loc[1] - target_center[1])
            
            if dist < min_dist:
                min_dist = dist
                best_move = (d, mtype)
        
        return best_move
