# search.py
"""
Alpha-beta search with iterative deepening and time management.
"""

from __future__ import annotations
from typing import Callable, Any, List, Tuple, Optional
import time
import math
import numpy as np
from game import board, enums
from .evaluation import Evaluator
from .trapdoor_belief import TrapdoorBelief


class SearchEngine:
    """
    Implements depth-limited alpha-beta search with iterative deepening.
    Manages time budgets to avoid timeouts.
    """
    
    def __init__(
        self, 
        evaluator: Evaluator, 
        trap_belief: TrapdoorBelief, 
        rng: np.random.Generator
    ):
        """
        Initialize the search engine.
        
        Args:
            evaluator: Evaluation function
            trap_belief: Trapdoor belief tracker
            rng: Random number generator for tie-breaking
        """
        self.evaluator = evaluator
        self.trap_belief = trap_belief
        self.rng = rng
        
        # Configuration knobs (can be tuned)
        self.max_search_depth = 6      # Absolute max depth
        self.min_depth = 1             # Starting depth for iterative deepening
        self.time_safety_margin = 2.0  # Stop searching if time_left() < this (seconds)
        self.nodes_searched = 0        # Statistics
        self.max_depth_reached = 0     # Statistics
    
    def choose_move(
        self, 
        game_board: board.Board, 
        time_left: Callable[[], float]
    ) -> Optional[Tuple[enums.Direction, enums.MoveType]]:
        """
        Iterative deepening alpha-beta search rooted at the current board.
        Returns one of board.get_valid_moves().
        
        Args:
            game_board: Current board state
            time_left: Function returning remaining time in seconds
            
        Returns:
            Best move as (Direction, MoveType) tuple, or None if no moves
        """
        valid_moves = game_board.get_valid_moves()
        
        if not valid_moves:
            # No moves available
            return None
        
        # Single move: no need to search
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Fallback: random valid move
        best_move = self.rng.choice(valid_moves)
        
        remaining_global = time_left()
        
        # Calculate per-move budget
        # Be conservative: use only a fraction of remaining time
        turns_remaining_estimate = max(1, game_board.turns_left_player)
        per_move_budget = max(0.05, (remaining_global - self.time_safety_margin) / (turns_remaining_estimate * 1.5))
        
        # Cap per-move budget to avoid spending too much on one move
        per_move_budget = min(per_move_budget, remaining_global / 5.0)
        
        # If we're extremely low on time, skip search
        if remaining_global < self.time_safety_margin:
            return self._choose_move_greedy(game_board, valid_moves)
        
        start_time = time.time()
        depth = self.min_depth
        self.nodes_searched = 0
        self.max_depth_reached = 0
        
        # Iterative deepening loop
        while depth <= self.max_search_depth:
            elapsed = time.time() - start_time
            
            # Check time budget
            if elapsed > per_move_budget or time_left() < self.time_safety_margin:
                break
            
            try:
                move_at_depth, value_at_depth = self._search_root(
                    game_board, depth, time_left, start_time, per_move_budget
                )
                
                if move_at_depth is not None:
                    best_move = move_at_depth
                    self.max_depth_reached = depth
                
                # If we found a guaranteed win, no need to search deeper
                if value_at_depth >= 5000:
                    break
                
            except TimeoutError:
                # Time ran out during search, use best move from previous depth
                break
            
            depth += 1
        
        return best_move
    
    def _choose_move_greedy(
        self, 
        game_board: board.Board, 
        valid_moves: List[Tuple[enums.Direction, enums.MoveType]]
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Simple fallback: simulate each move once, evaluate with heuristic, pick best.
        Used when time is nearly exhausted.
        
        Args:
            game_board: Current board state
            valid_moves: List of valid moves
            
        Returns:
            Best move according to shallow evaluation
        """
        best_move = valid_moves[0]
        best_val = -math.inf
        
        for move in valid_moves:
            child_board = self._simulate_move(game_board, move, is_opponent=False)
            if child_board is not None:
                val = self.evaluator.heuristic(child_board)
                if val > best_val:
                    best_val = val
                    best_move = move
        
        return best_move
    
    def _search_root(
        self, 
        game_board: board.Board, 
        depth: int, 
        time_left: Callable[[], float],
        start_time: float,
        time_budget: float
    ) -> Tuple[Optional[Tuple[enums.Direction, enums.MoveType]], float]:
        """
        Run alpha-beta search from the root (maximizing node).
        
        Args:
            game_board: Current board state
            depth: Search depth
            time_left: Function returning remaining time
            start_time: When this move search started
            time_budget: Time budget for this move
            
        Returns:
            (best_move, best_value) tuple
        """
        alpha = -math.inf
        beta = math.inf
        best_move = None
        best_value = -math.inf
        
        valid_moves = game_board.get_valid_moves()
        if not valid_moves:
            return None, self.evaluator.evaluate(game_board)
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(game_board, valid_moves, is_max=True)
        
        for move in ordered_moves:
            # Check time
            if time.time() - start_time > time_budget or time_left() < self.time_safety_margin:
                break
            
            child_board = self._simulate_move(game_board, move, is_opponent=False)
            if child_board is None:
                continue
            
            value = self._min_value(child_board, depth - 1, alpha, beta, time_left)
            
            if value > best_value or best_move is None:
                best_value = value
                best_move = move
            
            alpha = max(alpha, best_value)
            
            if beta <= alpha:
                break  # Alpha-beta cutoff
        
        return best_move, best_value
    
    def _max_value(
        self, 
        game_board: board.Board, 
        depth: int, 
        alpha: float, 
        beta: float, 
        time_left: Callable[[], float]
    ) -> float:
        """
        Maximizing player (us).
        
        Args:
            game_board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            time_left: Function returning remaining time
            
        Returns:
            Best value for maximizing player
        """
        self.nodes_searched += 1
        
        # Terminal test
        if self._terminal_test(game_board, depth, time_left):
            return self.evaluator.evaluate(game_board)
        
        value = -math.inf
        valid_moves = game_board.get_valid_moves()
        
        if not valid_moves:
            # No moves: evaluate terminal position
            return self.evaluator.evaluate(game_board)
        
        # Order moves
        ordered_moves = self._order_moves(game_board, valid_moves, is_max=True)
        
        for move in ordered_moves:
            if time_left() < self.time_safety_margin:
                break
            
            child_board = self._simulate_move(game_board, move, is_opponent=False)
            if child_board is None:
                continue
            
            value = max(value, self._min_value(child_board, depth - 1, alpha, beta, time_left))
            
            if value >= beta:
                return value  # Beta cutoff
            
            alpha = max(alpha, value)
        
        return value
    
    def _min_value(
        self, 
        game_board: board.Board, 
        depth: int, 
        alpha: float, 
        beta: float, 
        time_left: Callable[[], float]
    ) -> float:
        """
        Minimizing player (opponent).
        
        Args:
            game_board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            time_left: Function returning remaining time
            
        Returns:
            Best value for minimizing player
        """
        self.nodes_searched += 1
        
        # Terminal test
        if self._terminal_test(game_board, depth, time_left):
            return self.evaluator.evaluate(game_board)
        
        value = math.inf
        
        # Get opponent's valid moves by reversing perspective
        opp_board = game_board.get_copy()
        opp_board.reverse_perspective()
        valid_moves = opp_board.get_valid_moves()
        
        if not valid_moves:
            # Opponent has no moves: evaluate from our perspective
            return self.evaluator.evaluate(game_board)
        
        # Order moves
        ordered_moves = self._order_moves(opp_board, valid_moves, is_max=False)
        
        for move in ordered_moves:
            if time_left() < self.time_safety_margin:
                break
            
            child_board = self._simulate_move(opp_board, move, is_opponent=True)
            if child_board is None:
                continue
            
            value = min(value, self._max_value(child_board, depth - 1, alpha, beta, time_left))
            
            if value <= alpha:
                return value  # Alpha cutoff
            
            beta = min(beta, value)
        
        return value
    
    def _terminal_test(
        self, 
        game_board: board.Board, 
        depth: int, 
        time_left: Callable[[], float]
    ) -> bool:
        """
        Return True if we should stop searching.
        
        Args:
            game_board: Current board state
            depth: Remaining depth
            time_left: Function returning remaining time
            
        Returns:
            True if search should terminate
        """
        # Depth limit reached
        if depth <= 0:
            return True
        
        # Game is over
        if game_board.is_game_over():
            return True
        
        # Time is nearly exhausted
        if time_left() < self.time_safety_margin:
            return True
        
        return False
    
    def _simulate_move(
        self, 
        game_board: board.Board, 
        move: Tuple[enums.Direction, enums.MoveType], 
        is_opponent: bool
    ) -> Optional[board.Board]:
        """
        Return a new Board representing the result of applying move.
        
        Args:
            game_board: Current board state
            move: Move to apply (direction, move_type)
            is_opponent: Whether this is an opponent move
            
        Returns:
            New board state after move, or None if move is invalid
        """
        direction, move_type = move
        
        # Use the board's forecast_move method
        child_board = game_board.forecast_move(direction, move_type)
        
        if child_board is None:
            return None
        
        # If this is an opponent move, we need to reverse perspective back
        # because forecast_move switches perspective
        if is_opponent:
            child_board.reverse_perspective()
        
        return child_board
    
    def _order_moves(
        self, 
        game_board: board.Board, 
        moves: List[Tuple[enums.Direction, enums.MoveType]], 
        is_max: bool
    ) -> List[Tuple[enums.Direction, enums.MoveType]]:
        """
        Order moves for better alpha-beta pruning.
        
        Strategy:
        - Evaluate each move with a shallow heuristic
        - For max nodes: sort descending (best first)
        - For min nodes: sort ascending (worst first from opponent's view)
        
        Args:
            game_board: Current board state
            moves: List of valid moves
            is_max: Whether this is a maximizing node
            
        Returns:
            Ordered list of moves
        """
        if len(moves) <= 1:
            return moves
        
        # Score each move
        move_scores = []
        for move in moves:
            direction, move_type = move
            
            # Quick heuristic scoring
            score = 0.0
            
            # Prefer egg moves
            if move_type == enums.MoveType.EGG:
                score += 10.0
                # Extra bonus for corner eggs
                my_pos = game_board.chicken_player.get_location()
                next_pos = game_board.chicken_player.get_next_loc(direction)
                if next_pos is not None:
                    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
                    if my_pos in corners:
                        score += 20.0
            
            # Turd moves are situational
            elif move_type == enums.MoveType.TURD:
                score += 3.0
            
            # Prefer moves toward center in early game
            my_pos = game_board.chicken_player.get_location()
            next_pos = game_board.chicken_player.get_next_loc(direction, my_pos)
            if next_pos is not None:
                center = (3.5, 3.5)
                center_dist = abs(next_pos[0] - center[0]) + abs(next_pos[1] - center[1])
                score -= center_dist * 0.5
            
            # Penalize high-risk moves
            if next_pos is not None:
                risk = self.trap_belief.risk(next_pos)
                score -= risk * 50.0
            
            move_scores.append((move, score))
        
        # Sort based on whether this is max or min node
        move_scores.sort(key=lambda x: x[1], reverse=is_max)
        
        return [move for move, score in move_scores]

