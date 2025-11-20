from collections.abc import Callable
from typing import List, Tuple, Optional
import time

from game import *

"""
MinimaxAgent: Implements Iterative Deepening Minimax with Alpha-Beta Pruning
with Dynamic Time Budgeting and Move Ordering
"""


class PlayerAgent:
    """
    Optimized Minimax agent with alpha-beta pruning, iterative deepening,
    dynamic time budgeting, and move ordering.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        """
        Initialize the agent. Can be used for any pre-game setup.
        
        Args:
            board: Initial game board state
            time_left: Callable that returns remaining time in seconds
        """
        self.time_safety_threshold = 0.5  # Abort search if time drops below this
        self.max_depth_reached = 0
        self.sensor_data = None  # Store current sensor data for evaluation
        self.move_start_time = 0  # Track when current move started
        
    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Select the best move using iterative deepening minimax with alpha-beta pruning.
        
        Args:
            board: Current game state
            sensor_data: Trapdoor sensor data
            time_left: Callable returning remaining time in seconds
            
        Returns:
            Tuple of (Direction, MoveType) representing the best move
        """
        # Store sensor data for use in evaluation
        self.sensor_data = sensor_data
        
        # Start timing this move
        self.move_start_time = time.perf_counter()
        
        valid_moves = board.get_valid_moves()
        
        # Fallback: if only one move available or time is critical
        if len(valid_moves) == 1 or time_left() < self.time_safety_threshold:
            return valid_moves[0]
        
        # Calculate dynamic time budget for this move
        time_budget = self.calculate_time_budget(board, time_left())
        
        best_move = valid_moves[0]  # Default to first valid move
        depth = 1
        
        # Iterative deepening: try increasing depths until time budget exhausted
        while True:
            # Check if we've exceeded our time budget
            elapsed = time.perf_counter() - self.move_start_time
            if elapsed >= time_budget:
                break
            
            # Safety check: never let total time drop too low
            if time_left() < self.time_safety_threshold:
                break
            
            try:
                # Perform minimax search at current depth
                move, score = self.minimax_search(board, depth, time_left, time_budget)
                
                if move is not None:
                    best_move = move
                    self.max_depth_reached = depth
                    
                # If we found a guaranteed win, no need to search deeper
                if score == float('inf'):
                    break
                    
                depth += 1
                
                # Stop if we've searched very deep (likely won't complete next depth)
                if depth > 20:
                    break
                    
            except TimeoutError:
                # Time ran out during search, use best move from previous depth
                break
        
        return best_move
    
    def calculate_time_budget(self, board: board.Board, time_remaining: float) -> float:
        """
        Calculate how much time to allocate for this move.
        
        Args:
            board: Current game state
            time_remaining: Total time remaining in seconds
            
        Returns:
            Time budget for this move in seconds
        """
        # Estimate turns remaining (each player gets up to 40 turns)
        turns_played = board.turn_count // 2  # Divide by 2 since both players alternate
        estimated_turns_remaining = max(1, 40 - turns_played)
        
        # Allocate time: remaining_time / estimated_turns_remaining
        # But never allocate more than 15 seconds or less than 1 second per turn
        time_budget = time_remaining / estimated_turns_remaining
        time_budget = min(15.0, max(1.0, time_budget))
        
        # Reserve some safety margin
        time_budget = time_budget * 0.9
        
        return time_budget
    
    def minimax_search(
        self,
        board: board.Board,
        max_depth: int,
        time_left: Callable,
        time_budget: float
    ) -> Tuple[Optional[Tuple[enums.Direction, enums.MoveType]], float]:
        """
        Perform minimax search with alpha-beta pruning at the specified depth.
        
        Args:
            board: Current game state
            max_depth: Maximum depth to search
            time_left: Callable returning remaining time
            time_budget: Time budget for this move
            
        Returns:
            Tuple of (best_move, best_score)
        """
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            return None, float('-inf')
        
        # ORDER MOVES: Sort moves by heuristic evaluation (best first for MAX player)
        ordered_moves = self.order_moves(board, valid_moves, maximize=True)
        
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in ordered_moves:
            # Check time budget
            elapsed = time.perf_counter() - self.move_start_time
            if elapsed >= time_budget or time_left() < self.time_safety_threshold:
                raise TimeoutError("Time limit approaching")
            
            # Forecast the move and switch perspective to opponent
            new_board = board.forecast_move(move[0], move[1])
            
            if new_board is None:
                continue
            
            # Switch to opponent's perspective
            new_board.reverse_perspective()
            
            # Call min_value for opponent's turn
            score = self.min_value(new_board, max_depth - 1, alpha, beta, time_left, time_budget)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
            
            # Early exit if we found a guaranteed win
            if best_score == float('inf'):
                break
        
        return best_move, best_score
    
    def order_moves(
        self,
        board: board.Board,
        moves: List[Tuple[enums.Direction, enums.MoveType]],
        maximize: bool = True
    ) -> List[Tuple[enums.Direction, enums.MoveType]]:
        """
        Order moves by their heuristic evaluation to improve alpha-beta pruning.
        
        Args:
            board: Current game state
            moves: List of valid moves
            maximize: True if ordering for MAX player, False for MIN player
            
        Returns:
            Sorted list of moves (best moves first)
        """
        if len(moves) <= 1:
            return moves
        
        move_scores = []
        
        for move in moves:
            # Quickly evaluate this move
            new_board = board.forecast_move(move[0], move[1])
            if new_board is None:
                move_scores.append((move, float('-inf') if maximize else float('inf')))
                continue
            
            # Evaluate from current perspective (before reversing)
            score = self.quick_evaluate(new_board, maximize)
            move_scores.append((move, score))
        
        # Sort: descending for MAX (best first), ascending for MIN (worst for opponent first)
        move_scores.sort(key=lambda x: x[1], reverse=maximize)
        
        return [move for move, score in move_scores]
    
    def quick_evaluate(self, board: board.Board, from_max_perspective: bool) -> float:
        """
        Quick evaluation for move ordering (simpler than full evaluate).
        
        Args:
            board: Game state to evaluate
            from_max_perspective: True if evaluating from MAX player's view
            
        Returns:
            Quick heuristic score
        """
        if board.is_game_over():
            winner = board.get_winner()
            if winner == enums.Result.PLAYER:
                return float('inf') if from_max_perspective else float('-inf')
            elif winner == enums.Result.ENEMY:
                return float('-inf') if from_max_perspective else float('inf')
            return 0.0
        
        # Simple egg difference
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        
        if from_max_perspective:
            return my_eggs - enemy_eggs
        else:
            return enemy_eggs - my_eggs
    
    def max_value(
        self,
        board: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        time_left: Callable,
        time_budget: float
    ) -> float:
        """
        Maximizing player's turn in minimax algorithm.
        
        Args:
            board: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            time_left: Callable returning remaining time
            time_budget: Time budget for this move
            
        Returns:
            Best score for maximizing player
        """
        # Check time constraint
        elapsed = time.perf_counter() - self.move_start_time
        if elapsed >= time_budget or time_left() < self.time_safety_threshold:
            raise TimeoutError("Time limit approaching")
        
        # Terminal conditions
        if board.is_game_over():
            return self.evaluate(board)
        
        if depth <= 0:
            return self.evaluate(board)
        
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            return self.evaluate(board)
        
        # ORDER MOVES for better pruning
        ordered_moves = self.order_moves(board, valid_moves, maximize=True)
        
        max_score = float('-inf')
        
        for move in ordered_moves:
            # Forecast move and switch perspective
            new_board = board.forecast_move(move[0], move[1])
            
            if new_board is None:
                continue
            
            new_board.reverse_perspective()
            
            # Recursively call min_value
            score = self.min_value(new_board, depth - 1, alpha, beta, time_left, time_budget)
            
            max_score = max(max_score, score)
            alpha = max(alpha, max_score)
            
            # Alpha-beta pruning
            if max_score >= beta:
                break
        
        return max_score
    
    def min_value(
        self,
        board: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        time_left: Callable,
        time_budget: float
    ) -> float:
        """
        Minimizing player's turn in minimax algorithm.
        
        Args:
            board: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            time_left: Callable returning remaining time
            time_budget: Time budget for this move
            
        Returns:
            Best score for minimizing player
        """
        # Check time constraint
        elapsed = time.perf_counter() - self.move_start_time
        if elapsed >= time_budget or time_left() < self.time_safety_threshold:
            raise TimeoutError("Time limit approaching")
        
        # Terminal conditions
        if board.is_game_over():
            return self.evaluate(board)
        
        if depth <= 0:
            return self.evaluate(board)
        
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            return self.evaluate(board)
        
        # ORDER MOVES for better pruning (worst for opponent = best for us)
        ordered_moves = self.order_moves(board, valid_moves, maximize=False)
        
        min_score = float('inf')
        
        for move in ordered_moves:
            # Forecast move and switch perspective
            new_board = board.forecast_move(move[0], move[1])
            
            if new_board is None:
                continue
            
            new_board.reverse_perspective()
            
            # Recursively call max_value
            score = self.max_value(new_board, depth - 1, alpha, beta, time_left, time_budget)
            
            min_score = min(min_score, score)
            beta = min(beta, min_score)
            
            # Alpha-beta pruning
            if min_score <= alpha:
                break
        
        return min_score
    
    def evaluate(self, board: board.Board) -> float:
        """
        Evaluate the board state from the current player's perspective.
        
        Args:
            board: Game state to evaluate
            
        Returns:
            Heuristic score (higher is better for current player)
        """
        # Handle terminal states
        if board.is_game_over():
            winner = board.get_winner()
            if winner == enums.Result.PLAYER:
                return float('inf')
            elif winner == enums.Result.ENEMY:
                return float('-inf')
            else:  # Tie
                return 0.0
        
        # Primary metric: egg difference
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        score = float(my_eggs - enemy_eggs)
        
        # Get locations
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        
        # Corner bonus (corners give 3 eggs instead of 1)
        corners = [
            (0, 0),
            (0, board.game_map.MAP_SIZE - 1),
            (board.game_map.MAP_SIZE - 1, 0),
            (board.game_map.MAP_SIZE - 1, board.game_map.MAP_SIZE - 1)
        ]
        
        if my_loc in corners and board.can_lay_egg():
            score += 1.0
        
        if enemy_loc in corners:
            enemy_board = board.get_copy()
            enemy_board.reverse_perspective()
            if enemy_board.can_lay_egg():
                score -= 1.0
        
        # Trapdoor risk assessment
        score -= self.evaluate_trapdoor_risk(board, my_loc)
        
        # Mobility bonus (having more valid moves is advantageous)
        my_moves = len(board.get_valid_moves())
        if my_moves == 0:
            score -= 10.0  # Being blocked is very bad
        else:
            score += my_moves * 0.1
        
        # Enemy mobility
        enemy_board = board.get_copy()
        enemy_board.reverse_perspective()
        enemy_moves = len(enemy_board.get_valid_moves())
        if enemy_moves == 0:
            score += 10.0  # Blocking enemy is very good
        else:
            score -= enemy_moves * 0.1
        
        # Turd resource bonus
        my_turds = board.chicken_player.get_turds_left()
        enemy_turds = board.chicken_enemy.get_turds_left()
        score += (my_turds - enemy_turds) * 0.3
        
        # Positional advantage: being closer to center can be strategic
        center = board.game_map.MAP_SIZE / 2.0
        my_dist_to_center = abs(my_loc[0] - center) + abs(my_loc[1] - center)
        enemy_dist_to_center = abs(enemy_loc[0] - center) + abs(enemy_loc[1] - center)
        score += (enemy_dist_to_center - my_dist_to_center) * 0.05
        
        return score
    
    def evaluate_trapdoor_risk(self, board: board.Board, location: Tuple[int, int]) -> float:
        """
        Evaluate trapdoor risk at a given location based on sensor data.
        
        Args:
            board: Current game state
            location: Location to evaluate
            
        Returns:
            Risk penalty (higher = more risky)
        """
        if self.sensor_data is None:
            return 0.0
        
        risk = 0.0
        
        # Check each trapdoor's sensor data
        for i, (heard, felt) in enumerate(self.sensor_data):
            # If we've already found this trapdoor, no risk
            # (We can't directly check which trapdoor is which, so we estimate)
            
            # Strong signals indicate high trapdoor probability nearby
            if heard and felt:
                # Very close to a trapdoor - high risk
                risk += 3.0
            elif heard or felt:
                # Moderate proximity to trapdoor
                risk += 1.5
        
        # Penalize being on known trapdoors heavily
        if location in board.found_trapdoors:
            risk += 5.0
        
        return risk
