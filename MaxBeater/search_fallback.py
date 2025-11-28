"""
search_fallback.py - Fast fallback search for low-time situations

Provides lightweight search alternatives when time is critically low:
1. Greedy one-ply search (evaluate immediate moves)
2. Simple two-ply with enemy response
3. Move ordering for faster decision-making
"""

from __future__ import annotations
from typing import Callable, List, Tuple, Optional
import math
import numpy as np
from game import board, enums
from .evaluator import Evaluator
from .belief import TrapdoorBelief


class FallbackSearch:
    """
    Fast search engine for time-critical situations.
    
    Uses:
    - Greedy one-ply evaluation (fastest)
    - Optional two-ply with simple enemy response modeling
    - Quick heuristic evaluation only
    """
    
    def __init__(
        self,
        evaluator: Evaluator,
        trap_belief: TrapdoorBelief,
        rng: np.random.Generator
    ):
        """
        Initialize fallback search engine.
        
        Args:
            evaluator: Board evaluator
            trap_belief: Trapdoor belief tracker
            rng: Random number generator
        """
        self.evaluator = evaluator
        self.trap_belief = trap_belief
        self.rng = rng
    
    def _filter_moves_early_game(
        self,
        game_board: board.Board,
        moves: List[Tuple[enums.Direction, enums.MoveType]]
    ) -> List[Tuple[enums.Direction, enums.MoveType]]:
        """
        Filter early-game moves to avoid wasting TURDs when no eggs are on the board.
        """
        if not moves:
            return moves
        
        my_eggs = game_board.chicken_player.get_eggs_laid()
        enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
        total_eggs = my_eggs + enemy_eggs
        
        turns_left = getattr(game_board, "turns_left_player", 40)
        
        # Early game: no eggs yet and plenty of turns left.
        if total_eggs == 0 and turns_left > 20:
            non_turd_moves = [
                m for m in moves
                if m[1] != enums.MoveType.TURD
            ]
            # Only filter if we have at least one non-TURD move.
            if non_turd_moves:
                return non_turd_moves
        
        return moves
    
    def choose_move(
        self,
        game_board: board.Board,
        time_left: Callable[[], float]
    ) -> Optional[Tuple[enums.Direction, enums.MoveType]]:
        """
        Choose move using fast greedy search.
        
        Strategy:
        - If time > 2s: two-ply search (our move + enemy response)
        - If time < 2s: one-ply greedy (immediate evaluation only)
        
        Args:
            game_board: Current board state
            time_left: Function returning remaining time
            
        Returns:
            Best move found
        """
        valid_moves = game_board.get_valid_moves(enemy=False)
        
        # Filter out bad early-game moves (e.g., TURD spam)
        valid_moves = self._filter_moves_early_game(game_board, valid_moves)
        
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        remaining_time = time_left()
        
        if remaining_time > 2.0:
            # Enough time for two-ply
            return self._two_ply_search(game_board, valid_moves, time_left)
        else:
            # Very low time: one-ply greedy only
            return self._one_ply_greedy(game_board, valid_moves)
    
    def _one_ply_greedy(
        self,
        game_board: board.Board,
        valid_moves: List[Tuple[enums.Direction, enums.MoveType]]
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Greedy one-ply search: evaluate immediate position after each move.
        
        Args:
            game_board: Current board
            valid_moves: List of valid moves
            
        Returns:
            Move with highest immediate evaluation
        """
        best_move = valid_moves[0]
        best_score = -math.inf
        
        for move in valid_moves:
            # Apply move
            child_board = self._apply_move(game_board, move)
            
            if child_board is None:
                continue
            
            # Quick evaluation (heuristic only, no value model for speed)
            # Note: child_board is from opponent's perspective after reverse_perspective
            # So we negate the evaluation
            score = -self.evaluator.quick_evaluate(child_board)
            
            # Move-type biasing: encourage eggs, discourage early turds
            direction, move_type = move
            
            my_eggs = game_board.chicken_player.get_eggs_laid()
            enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
            total_eggs = my_eggs + enemy_eggs
            turns_left = getattr(game_board, "turns_left_player", 40)
            
            # Small positive bonus for laying eggs, especially earlier in the game
            if move_type == enums.MoveType.EGG:
                if turns_left > 10:
                    score += 150.0
                else:
                    score += 100.0
            
            # Penalty for wasting turds when no eggs are on the board and still early
            if move_type == enums.MoveType.TURD and total_eggs == 0 and turns_left > 20:
                score -= 150.0
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _two_ply_search(
        self,
        game_board: board.Board,
        valid_moves: List[Tuple[enums.Direction, enums.MoveType]],
        time_left: Callable[[], float]
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Two-ply search: our move + enemy's best response.
        
        For each of our moves:
        1. Apply our move
        2. Find enemy's best counter-move
        3. Evaluate resulting position
        
        Args:
            game_board: Current board
            valid_moves: Our valid moves
            time_left: Time checker
            
        Returns:
            Move that maximizes our score after enemy response
        """
        best_move = valid_moves[0]
        best_score = -math.inf
        
        for our_move in valid_moves:
            # Apply our move
            after_our_move = self._apply_move(game_board, our_move)
            
            if after_our_move is None:
                continue
            
            # Now after_our_move represents enemy's perspective
            enemy_moves = after_our_move.get_valid_moves(enemy=False)
            
            if not enemy_moves:
                # Enemy has no moves: excellent for us!
                # They get blocked, we get +5 eggs bonus
                score = 1000000.0
            else:
                # Find enemy's best response (worst for us)
                worst_enemy_score = math.inf
                
                for enemy_move in enemy_moves:
                    # Check time periodically
                    if time_left() < 1.0:
                        break
                    
                    # Apply enemy move
                    after_enemy_move = self._apply_move(after_our_move, enemy_move)
                    
                    if after_enemy_move is None:
                        continue
                    
                    # Evaluate from our perspective (after enemy move, it's our turn again)
                    # after_enemy_move is from our perspective after second reverse
                    enemy_eval = -self.evaluator.quick_evaluate(after_enemy_move)
                    
                    # Enemy picks move that minimizes our score
                    worst_enemy_score = min(worst_enemy_score, enemy_eval)
                
                score = worst_enemy_score
            
            # Apply a small move-type bias for our move
            direction, move_type = our_move
            
            my_eggs = game_board.chicken_player.get_eggs_laid()
            enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
            total_eggs = my_eggs + enemy_eggs
            turns_left = getattr(game_board, "turns_left_player", 40)
            
            if move_type == enums.MoveType.EGG:
                if turns_left > 10:
                    score += 150.0
                else:
                    score += 100.0
            
            if move_type == enums.MoveType.TURD and total_eggs == 0 and turns_left > 20:
                score -= 150.0
            
            # Pick our move that maximizes score after enemy response
            if score > best_score:
                best_score = score
                best_move = our_move
        
        return best_move
    
    def _apply_move(
        self,
        game_board: board.Board,
        move: Tuple[enums.Direction, enums.MoveType]
    ) -> Optional[board.Board]:
        """
        Apply a move and return new board with perspective reversed.
        
        Same logic as in MCTS: forecast_move + reverse_perspective.
        
        Args:
            game_board: Current board
            move: Move to apply
            
        Returns:
            New board from opponent's perspective, or None if invalid
        """
        try:
            direction, move_type = move
            
            # Clone and apply move
            new_board = game_board.forecast_move(direction, move_type, check_ok=True)
            
            if new_board is None:
                return None
            
            # Reverse perspective for opponent's turn
            new_board.reverse_perspective()
            
            return new_board
            
        except Exception as e:
            return None

