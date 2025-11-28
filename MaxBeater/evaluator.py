"""
evaluator.py - Board evaluation combining heuristics and learned value function

Provides comprehensive evaluation from current player's perspective.
Combines hand-crafted heuristics with optional neural network value estimation.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from game import board, enums
from .belief import TrapdoorBelief
from .features import extract_scalar_features, encode_board_tensor
from .value_model_runtime import ValueModelRuntime


class Evaluator:
    """
    Evaluates board states from the current player's perspective.
    
    Combines:
    1. Hand-crafted heuristic (always available)
    2. Learned value function (if weights available)
    
    The heuristic focuses on:
    - Egg differential (primary objective)
    - Mobility and blocking
    - Strategic positioning (corners, center)
    - Trapdoor risk management
    - Turd resource management
    """
    
    def __init__(self, trap_belief: TrapdoorBelief):
        """
        Initialize evaluator.
        
        Args:
            trap_belief: Trapdoor belief tracker (shared with agent)
        """
        self.trap_belief = trap_belief
        self.value_model = ValueModelRuntime()
        
        # Heuristic weights (tuned for importance)
        self.weights = {
            'egg_diff': 1000.0,          # Most important: egg differential
            'corner_egg_bonus': 200.0,   # Corners give 3x eggs
            'mobility': 15.0,            # Having more valid moves
            'blocking_bonus': 500.0,     # Blocking enemy completely
            'blocked_penalty': 500.0,    # Being blocked is very bad
            'trap_risk_current': 150.0,  # Risk at current position
            'trap_risk_nearby': 50.0,    # Risk in neighborhood
            'turd_diff': 30.0,           # Turd resource advantage
            'center_control': 10.0,      # Central positioning
            'endgame_egg_mult': 1.5,     # Multiply egg importance in endgame
        }
    
    def heuristic(self, game_board: board.Board) -> float:
        """
        Hand-crafted heuristic evaluation.
        
        Returns:
            Score from current player's perspective (higher = better)
        """
        # Check for terminal states first
        if game_board.is_game_over():
            return self._terminal_value(game_board)
        
        score = 0.0
        
        # === PRIMARY: Egg differential ===
        my_eggs = game_board.chicken_player.get_eggs_laid()
        enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
        egg_diff = my_eggs - enemy_eggs
        
        # Increase egg importance in endgame
        turns_left = game_board.turns_left_player
        if turns_left < 10:
            egg_weight = self.weights['egg_diff'] * self.weights['endgame_egg_mult']
        else:
            egg_weight = self.weights['egg_diff']
        
        score += egg_diff * egg_weight
        
        # === MOBILITY: Valid moves ===
        my_moves = game_board.get_valid_moves(enemy=False)
        num_my_moves = len(my_moves)
        
        # Create enemy perspective board to check their mobility
        enemy_board = game_board.get_copy()
        enemy_board.reverse_perspective()
        enemy_moves = enemy_board.get_valid_moves(enemy=False)
        num_enemy_moves = len(enemy_moves)
        
        # Being blocked is catastrophic (enemy gets +5 eggs)
        if num_my_moves == 0:
            score -= self.weights['blocked_penalty']
        else:
            score += num_my_moves * self.weights['mobility']
        
        # Blocking enemy is excellent (we get +5 eggs)
        if num_enemy_moves == 0:
            score += self.weights['blocking_bonus']
        else:
            score -= num_enemy_moves * self.weights['mobility']
        
        # === STRATEGIC POSITIONING ===
        my_pos = game_board.chicken_player.get_location()
        enemy_pos = game_board.chicken_enemy.get_location()
        
        # Corner bonus: corners give 3 eggs instead of 1
        corners = [
            (0, 0),
            (0, game_board.game_map.MAP_SIZE - 1),
            (game_board.game_map.MAP_SIZE - 1, 0),
            (game_board.game_map.MAP_SIZE - 1, game_board.game_map.MAP_SIZE - 1)
        ]
        
        if my_pos in corners and game_board.can_lay_egg():
            score += self.weights['corner_egg_bonus']
        
        if enemy_pos in corners and enemy_board.can_lay_egg():
            score -= self.weights['corner_egg_bonus']
        
        # Center control (more strategic in midgame)
        center = game_board.game_map.MAP_SIZE / 2.0
        my_dist_center = abs(my_pos[0] - center) + abs(my_pos[1] - center)
        enemy_dist_center = abs(enemy_pos[0] - center) + abs(enemy_pos[1] - center)
        
        # Prefer center early/mid game
        if turns_left > 15:
            score += (enemy_dist_center - my_dist_center) * self.weights['center_control']
        
        # === TRAPDOOR RISK MANAGEMENT ===
        my_risk = self.trap_belief.risk(my_pos)
        my_nearby_risk = self.trap_belief.max_risk_in_radius(my_pos, radius=2)
        
        score -= my_risk * self.weights['trap_risk_current']
        score -= my_nearby_risk * self.weights['trap_risk_nearby']
        
        # Enemy trap risk is good for us
        enemy_risk = self.trap_belief.risk(enemy_pos)
        score += enemy_risk * self.weights['trap_risk_current'] * 0.5
        
        # === TURD RESOURCE MANAGEMENT ===
        my_turds = game_board.chicken_player.get_turds_left()
        enemy_turds = game_board.chicken_enemy.get_turds_left()
        turd_diff = my_turds - enemy_turds
        
        score += turd_diff * self.weights['turd_diff']
        
        return score
    
    def _terminal_value(self, game_board: board.Board) -> float:
        """
        Evaluate terminal (game over) states.
        
        Returns:
            Very large positive/negative value based on outcome
        """
        winner = game_board.get_winner()
        
        if winner == enums.Result.PLAYER:
            # We won
            return 1000000.0
        elif winner == enums.Result.ENEMY:
            # We lost
            return -1000000.0
        else:
            # Tie
            return 0.0
    
    def evaluate(self, game_board: board.Board) -> float:
        """
        Combined evaluation using heuristic + value model.
        
        Strategy:
        1. Always compute heuristic (fast and reliable).
        2. Use the value model only as a small correction in mid/late-game.
        3. Trust heuristic entirely in:
           - Terminal states
           - Very confident positions
           - Early game / no eggs yet
        
        Returns:
            Final evaluation score.
        """
        # Terminal states: return immediately
        if game_board.is_game_over():
            return self._terminal_value(game_board)
        
        # Always compute heuristic
        h = self.heuristic(game_board)
        
        # If value model not loaded, use heuristic only
        if not self.value_model.is_loaded():
            return h
        
        # Early-game gate: don't trust value net before eggs or when very early
        my_eggs = game_board.chicken_player.get_eggs_laid()
        enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
        total_eggs = my_eggs + enemy_eggs
        turns_left = getattr(game_board, "turns_left_player", 40)
        
        if total_eggs == 0 or turns_left > 30:
            return h
        
        # Use value net as a small correction in mid/late game
        try:
            board_tensor = encode_board_tensor(game_board, self.trap_belief)  # (14, 8, 8)
            scalars = extract_scalar_features(game_board, self.trap_belief)    # (24,)
            
            # Concatenate into single input vector
            x = np.concatenate([board_tensor.flatten(), scalars])
            
            # Get value model prediction (in [-1, 1])
            v = self.value_model.forward(x)
            
        except Exception as e:
            # If value model fails for any reason, fall back to heuristic
            print(f"[Evaluator] Value model error: {e}")
            return h
        
        # Scale value model output much more conservatively
        scaled_v = v * 300.0
        
        # Heuristic-dominant blending
        abs_h = abs(h)
        
        if abs_h > 2000:
            # Very confident heuristic: ignore value model
            return h
        elif abs_h > 1000:
            # Somewhat confident: 80% heuristic, 20% value
            alpha = 0.2
        else:
            # Uncertain position: 65% heuristic, 35% value
            alpha = 0.35
        
        return (1.0 - alpha) * h + alpha * scaled_v
    
    def quick_evaluate(self, game_board: board.Board) -> float:
        """
        Fast evaluation for move ordering (heuristic only).
        
        Args:
            game_board: Board state
            
        Returns:
            Quick heuristic score
        """
        if game_board.is_game_over():
            return self._terminal_value(game_board)
        
        # Simple evaluation: egg differential + mobility
        my_eggs = game_board.chicken_player.get_eggs_laid()
        enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
        
        my_moves = len(game_board.get_valid_moves(enemy=False))
        
        # Quick score
        score = (my_eggs - enemy_eggs) * 1000.0
        
        if my_moves == 0:
            score -= 500.0
        else:
            score += my_moves * 10.0
        
        return score

