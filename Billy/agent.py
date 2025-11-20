# agent.py
"""
Bob - Advanced AI agent using adversarial search with alpha-beta pruning,
Bayesian trapdoor tracking, and learned value estimation.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import List, Tuple, Optional
import numpy as np
from game import board, enums

from .trapdoor_belief import TrapdoorBelief
from .evaluation import Evaluator
from .search import SearchEngine


class PlayerAgent:
    """
    Main agent class instantiated by the game engine.
    This class is intentionally thin and delegates most logic to helper classes.
    
    Architecture:
    - TrapdoorBelief: Bayesian belief tracking over trapdoor locations
    - Evaluator: Board evaluation using heuristics + optional learned value net
    - SearchEngine: Alpha-beta search with iterative deepening
    """

    def __init__(self, board: board.Board, time_left: Callable, seed: Optional[int] = None):
        """
        Initialize the agent and its components.
        
        Args:
            board: Initial game board state
            time_left: Callable that returns remaining time in seconds
            seed: Random seed for reproducibility (optional)
        """
        # Initialize RNG for tie-breaking and any randomization
        self.rng = np.random.default_rng(seed)
        
        # Trapdoor belief model (probabilistic reasoning)
        self.trap_belief = TrapdoorBelief(board_size=8)
        
        # Evaluator (heuristic + optional learned value net)
        self.evaluator = Evaluator(self.trap_belief)
        
        # Search/planning engine
        self.searcher = SearchEngine(
            evaluator=self.evaluator,
            trap_belief=self.trap_belief,
            rng=self.rng,
        )

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Called by the engine to choose a move for the current turn.
        
        Args:
            board: Current Board object with full game state
            sensor_data: Per-trapdoor tuple of (heard, felt) booleans
            time_left: Function returning remaining wall-clock time in seconds
            
        Returns:
            Move as (Direction, MoveType) tuple
        """
        # Update trapdoor beliefs using current sensor data
        self.trap_belief.update(board, sensor_data)
        
        # Ask search engine to choose a move
        move = self.searcher.choose_move(board, time_left)
        
        # Safety check: ensure move is in the list of valid moves
        valid_moves = board.get_valid_moves()
        
        if move is None or (move not in valid_moves):
            # Fallback: choose a random valid move
            if valid_moves:
                move = valid_moves[self.rng.integers(0, len(valid_moves))]
            else:
                # No valid moves available (should rarely happen)
                # Return a default move (will likely be rejected by engine)
                move = (enums.Direction.UP, enums.MoveType.PLAIN)
        
        return move
