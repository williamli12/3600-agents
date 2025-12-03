"""
Search module for MinimaxAgent.
Implements alpha-beta search with iterative deepening.
Simplified from initial complex version to match Bob's proven architecture.
"""

from typing import Tuple, List, Callable, Optional
import math

from game import board, enums
from .evaluation import Evaluator
from .trapdoor_belief import TrapdoorBelief


class SearchEngine:
    """
    Alpha-beta search with iterative deepening.
    Simplified to focus on speed and core minimax principles.
    """
    
    def __init__(self, evaluator: Evaluator, trap_belief: TrapdoorBelief):
        self.evaluator = evaluator
        self.trap_belief = trap_belief
        
        # Search configuration
        self.max_depth = 6              # Same as Bob
        self.min_depth = 1
        self.time_safety_margin = 2.0   # Same as Bob
        
        # Statistics
        self.nodes_searched = 0
        self.max_depth_reached = 0
    
    def search(
        self, 
        game_board: board.Board, 
        time_left: Callable[[], float]
    ) -> Optional[Tuple[enums.Direction, enums.MoveType]]:
        """
        Choose best move using iterative deepening alpha-beta search.
        
        Args:
            game_board: Current board state
            time_left: Function that returns remaining time in seconds
            
        Returns:
            Best move (direction, move_type) or None if no valid moves
        """
        # Reset statistics
        self.nodes_searched = 0
        self.max_depth_reached = 0
        
        valid_moves = game_board.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Calculate time budget
        remaining_time = time_left()
        turns_estimate = max(1, game_board.turns_left_player)
        per_move_budget = (remaining_time - self.time_safety_margin) / (turns_estimate * 1.5)
        per_move_budget = max(0.05, min(per_move_budget, remaining_time / 5.0))
        
        if remaining_time < self.time_safety_margin:
            return self._emergency_move(game_board, valid_moves)
        
        # Iterative deepening
        best_move = valid_moves[0]  # Fallback
        search_start_time = time_left()
        
        for depth in range(self.min_depth, self.max_depth + 1):
            if time_left() < self.time_safety_margin:
                break
            
            # Time check: don't start a new depth if we don't have enough time
            time_used = search_start_time - time_left()
            if time_used > per_move_budget * 0.7:
                break
            
            try:
                move, _ = self._minimax(
                    game_board,
                    depth,
                    -math.inf,
                    math.inf,
                    time_left
                )
                
                if move is not None:
                    best_move = move
                    self.max_depth_reached = depth
            
            except Exception:
                # If search fails, use last known best move
                break
        
        return best_move
    
    def _minimax(
        self,
        game_board: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        time_left: Callable[[], float]
    ) -> Tuple[Optional[Tuple[enums.Direction, enums.MoveType]], float]:
        """
        Root minimax call (maximizing player).
        
        Returns:
            (best_move, best_value)
        """
        valid_moves = game_board.get_valid_moves()
        
        if not valid_moves:
            return None, self.evaluator.evaluate(game_board)
        
        # Order moves
        ordered_moves = self._order_moves(game_board, valid_moves, is_max=True)
        
        best_move = ordered_moves[0]
        best_value = -math.inf
        
        for move in ordered_moves:
            if time_left() < self.time_safety_margin:
                break
            
            child_board = self._make_move(game_board, move)
            if child_board is None:
                continue
            
            value = self._min_value(child_board, depth - 1, alpha, beta, time_left)
            
            if value > best_value:
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
        """Maximizing player (us)."""
        self.nodes_searched += 1
        
        # Terminal test
        if self._terminal_test(game_board, depth, time_left):
            return self.evaluator.evaluate(game_board)
        
        value = -math.inf
        valid_moves = game_board.get_valid_moves()
        
        if not valid_moves:
            return self.evaluator.evaluate(game_board)
        
        # Order moves
        ordered_moves = self._order_moves(game_board, valid_moves, is_max=True)
        
        for move in ordered_moves:
            if time_left() < self.time_safety_margin:
                break
            
            child_board = self._make_move(game_board, move)
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
        """Minimizing player (opponent)."""
        self.nodes_searched += 1
        
        # Terminal test
        if self._terminal_test(game_board, depth, time_left):
            return self.evaluator.evaluate(game_board)
        
        value = math.inf
        
        # Get opponent's valid moves (already in opponent's perspective)
        valid_moves = game_board.get_valid_moves()
        
        if not valid_moves:
            return self.evaluator.evaluate(game_board)
        
        # Order moves from opponent's perspective
        ordered_moves = self._order_moves(game_board, valid_moves, is_max=False)
        
        for move in ordered_moves:
            if time_left() < self.time_safety_margin:
                break
            
            child_board = self._make_move(game_board, move)
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
        """Return True if we should stop searching."""
        if depth <= 0:
            return True
        
        if game_board.is_game_over():
            return True
        
        if time_left() < self.time_safety_margin:
            return True
        
        return False
    
    def _make_move(
        self,
        game_board: board.Board,
        move: Tuple[enums.Direction, enums.MoveType]
    ) -> Optional[board.Board]:
        """
        Apply move and return resulting board.
        CRITICAL: forecast_move switches perspective!
        """
        child = game_board.forecast_move(move[0], move[1])
        if child is None:
            return None
        
        # forecast_move switches turns but NOT perspective
        # Must reverse perspective so opponent becomes "player"
        child.reverse_perspective()
        
        return child
    
    def _order_moves(
        self, 
        game_board: board.Board, 
        moves: List[Tuple[enums.Direction, enums.MoveType]], 
        is_max: bool
    ) -> List[Tuple[enums.Direction, enums.MoveType]]:
        """
        Order moves for better alpha-beta pruning.
        Uses Bob's proven simple heuristic.
        """
        if len(moves) <= 1:
            return moves
        
        move_scores = []
        for move in moves:
            direction, move_type = move
            score = 0.0
            
            # Prefer egg moves
            if move_type == enums.MoveType.EGG:
                score += 10.0
                # Extra bonus for corner eggs
                my_pos = game_board.chicken_player.get_location()
                corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
                if my_pos in corners:
                    score += 20.0
            
            # Turd moves are situational
            elif move_type == enums.MoveType.TURD:
                score += 3.0
            
            # Prefer moves toward center
            my_pos = game_board.chicken_player.get_location()
            next_pos = game_board.chicken_player.get_next_loc(direction, my_pos)
            if next_pos is not None:
                center = (3.5, 3.5)
                center_dist = abs(next_pos[0] - center[0]) + abs(next_pos[1] - center[1])
                score -= center_dist * 0.5
                
                # CRITICAL: NEVER go to known trapdoors!
                if next_pos in game_board.found_trapdoors:
                    score -= 100000.0  # Absolutely avoid
                else:
                    # Aggressively penalize high-risk moves based on belief
                    risk = self.trap_belief.risk(next_pos)
                    score -= risk * 200.0  # Quadrupled from original 50
                    
                    # Extra penalty for moves to squares with high uncertainty
                    if risk > 0.15:
                        score -= 100.0  # Doubled penalty
                    if risk > 0.25:
                        score -= 200.0  # Extra for very risky squares
            
            move_scores.append((move, score))
        
        # Sort based on whether this is max or min node
        move_scores.sort(key=lambda x: x[1], reverse=is_max)
        
        return [move for move, score in move_scores]
    
    def _emergency_move(
        self, 
        game_board: board.Board, 
        valid_moves: List[Tuple[enums.Direction, enums.MoveType]]
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Emergency greedy move selection when time is critical.
        """
        # Try to lay egg if possible
        for move in valid_moves:
            if move[1] == enums.MoveType.EGG:
                my_pos = game_board.chicken_player.get_location()
                # Avoid known trapdoors!
                if my_pos not in game_board.found_trapdoors:
                    return move
        
        # Otherwise just pick first valid move
        return valid_moves[0]
