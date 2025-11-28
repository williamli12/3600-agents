from collections.abc import Callable
from typing import List, Tuple, Optional
import random

from game import board, enums

from .transposition import TranspositionTable
from .belief import TrapdoorBelief
from .evaluation import Evaluator
from .search import SearchEngine

class PlayerAgent:
    """
    Gary: The Super Agent.
    Uses Iterative Deepening PVS with Zobrist Hashing and Bayesian Trapdoor Tracking.
    """

    def __init__(self, board: board.Board, time_left: Callable, seed: Optional[int] = None):
        # Initialize Components
        self.tt = TranspositionTable(board_size=board.game_map.MAP_SIZE)
        self.belief = TrapdoorBelief(board_size=board.game_map.MAP_SIZE)
        self.visited_locations = set() # Track visited locations for exploration
        self.evaluator = Evaluator(self.belief, self.visited_locations)
        self.searcher = SearchEngine(self.evaluator, self.tt)
        
        # Seed RNG if needed (Gary components use random module)
        if seed is not None:
            random.seed(seed)
        
        # Pre-calculate something? No.

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Select the best move.
        """
        # Update visited locations
        self.visited_locations.add(board.chicken_player.get_location())
        
        # 1. Update Beliefs
        self.belief.update(board, sensor_data)
        
        # 2. Search for Best Move
        move = self.searcher.search(board, time_left)
        
        # 3. Fallback (Safety)
        if move is None:
            valid_moves = board.get_valid_moves()
            if valid_moves:
                move = random.choice(valid_moves)
            else:
                # No valid moves? Game likely over, but must return something to avoid crash
                move = (enums.Direction.UP, enums.MoveType.PLAIN)
        
        return move
