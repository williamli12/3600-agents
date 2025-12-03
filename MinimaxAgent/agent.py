"""
MinimaxAgent - Optimized agent using minimax with alpha-beta pruning.
Simplified architecture focusing on speed and effectiveness.
"""

from collections.abc import Callable
from typing import List, Tuple
import random
from game import board, enums

from .trapdoor_belief import TrapdoorBelief
from .evaluation import Evaluator
from .search import SearchEngine


class PlayerAgent:
    """
    MinimaxAgent - Fast and effective AI using alpha-beta search.
    
    Architecture:
    - Alpha-beta pruning with iterative deepening
    - Bayesian trapdoor tracking
    - Balanced evaluation function
    - Simple but effective move ordering
    """

    def __init__(self, initial_board: board.Board, time_left: Callable[[], float]):
        """Initialize the agent."""
        # Initialize components
        self.trap_belief = TrapdoorBelief(board_size=8)
        self.evaluator = Evaluator(self.trap_belief)
        self.searcher = SearchEngine(self.evaluator, self.trap_belief)

    def play(
        self,
        game_board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Choose a move for the current turn.
        
        Args:
            game_board: Current board state
            sensor_data: [(even_heard, even_felt), (odd_heard, odd_felt)]
            time_left: Function returning remaining time in seconds
            
        Returns:
            Move as (Direction, MoveType) tuple
        """
        # Update trapdoor beliefs
        self.trap_belief.update(game_board, sensor_data)
        
        # Search for best move
        move = self.searcher.search(game_board, time_left)
        
        # Safety check: ensure move is valid AND doesn't go to known trapdoor
        valid_moves = game_board.get_valid_moves()
        
        # CRITICAL: Filter out moves to known trapdoors
        safe_moves = []
        my_pos = game_board.chicken_player.get_location()
        for m in valid_moves:
            next_pos = game_board.chicken_player.get_next_loc(m[0], my_pos)
            if next_pos is not None and next_pos not in game_board.found_trapdoors:
                safe_moves.append(m)
        
        # Use safe moves if available, otherwise use all valid moves (trapped situation)
        moves_to_consider = safe_moves if safe_moves else valid_moves
        
        if move is None or (move not in moves_to_consider):
            # Fallback: choose safest available move
            if moves_to_consider:
                # Prefer egg moves if available and safe
                for m in moves_to_consider:
                    if m[1] == enums.MoveType.EGG:
                        pos = game_board.chicken_player.get_location()
                        # Don't lay egg on known trapdoor
                        if pos not in game_board.found_trapdoors:
                            move = m
                            break
                if move is None or move not in moves_to_consider:
                    # Pick move with lowest trapdoor risk
                    best_move = None
                    best_risk = float('inf')
                    for m in moves_to_consider:
                        next_pos = game_board.chicken_player.get_next_loc(m[0], my_pos)
                        if next_pos:
                            risk = self.trap_belief.risk(next_pos)
                            if risk < best_risk:
                                best_risk = risk
                                best_move = m
                    move = best_move if best_move else moves_to_consider[0]
            else:
                # No valid moves (shouldn't happen)
                move = (enums.Direction.UP, enums.MoveType.PLAIN)
        
        return move
