# evaluation.py
"""
Sophisticated board evaluation combining hand-crafted heuristics with strategic reasoning.
Evaluates positions from the current player's perspective.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from game import board, enums
from .trapdoor_belief import TrapdoorBelief
from .features import extract_features, quick_tactical_score


class Evaluator:
    """
    Evaluates board positions using sophisticated multi-layered heuristics.
    Balances material, positional, and tactical considerations.
    """
    
    def __init__(self, trap_belief: TrapdoorBelief):
        """
        Initialize the evaluator.
        
        Args:
            trap_belief: Trapdoor belief tracker for risk assessment
        """
        self.trap_belief = trap_belief
        
        # Heuristic weights (balanced like Bob, proven effective)
        self.weights = {
            'egg_diff': 100.0,            # Primary objective (Bob: 100)
            'egg_diff_endgame': 150.0,    # Slightly more in endgame
            'corner_egg_bonus': 35.0,     # Corner eggs worth 3x (Bob: 30)
            'blocking_bonus': 5000.0,     # Blocking opponent (+5 eggs)
            'blocked_penalty': 5000.0,    # Being blocked (-5 eggs for them)
            'mobility': 2.0,              # Value each move option (Bob: 2)
            'mobility_diff': 5.0,         # Mobility advantage (Bob: 5)
            'trapdoor_risk_pos': 800.0,   # Risk at current position (Bob: 800)
            'trapdoor_risk_nearby': 200.0,# Risk nearby (Bob: 200)
            'corner_proximity': 2.0,      # Being near corners
            'center_control_mid': 1.5,    # Center control in midgame (Bob: 1)
            'center_control_late': 0.5,   # Less important in endgame
            'turd_advantage': 3.0,        # Having turds available (Bob: 3)
            'territory_control': 1.0,     # Controlling board space
            'tempo_advantage': 5.0,       # Having more turns left (Bob: 5)
            'time_pressure': 0.5,         # Time remaining factor
            'can_lay_egg_bonus': 5.0,    # Being able to lay egg now
            'distance_to_opp': 3.0,       # Optimal blocking distance
        }
    
    def quick_eval(self, game_board: board.Board) -> float:
        """
        Ultra-fast evaluation for move ordering and time-critical situations.
        Uses simplified tactical score.
        
        Args:
            game_board: Board state to evaluate
            
        Returns:
            Quick evaluation score
        """
        return quick_tactical_score(game_board)
    
    def evaluate(self, game_board: board.Board) -> float:
        """
        Full sophisticated evaluation of a board position.
        
        Combines multiple strategic factors with game-phase awareness.
        
        Args:
            game_board: Board state to evaluate
            
        Returns:
            Evaluation score from current player's perspective (higher is better)
        """
        score = 0.0
        
        # === TERMINAL STATES (Most Important) ===
        if game_board.is_game_over():
            winner = game_board.get_winner()
            if winner == enums.Result.PLAYER:
                return 10000.0  # We won (Bob: 10000)
            elif winner == enums.Result.ENEMY:
                return -10000.0  # We lost (Bob: -10000)
            else:
                return 0.0  # Tie
        
        # === BLOCKING STATES (Critical) ===
        my_moves = game_board.get_valid_moves()
        if len(my_moves) == 0:
            # Being blocked is catastrophic (-5 eggs for opponent = +5 for them)
            score -= self.weights['blocked_penalty']
        
        # Check if opponent is blocked
        opp_board = game_board.get_copy()
        opp_board.reverse_perspective()
        opp_moves = opp_board.get_valid_moves()
        if len(opp_moves) == 0:
            # Blocking opponent is excellent (+5 eggs for us)
            score += self.weights['blocking_bonus']
        
        # === GAME PHASE DETECTION ===
        total_turns = game_board.turn_count
        my_turns_left = game_board.turns_left_player
        opp_turns_left = game_board.turns_left_enemy
        
        is_endgame = (my_turns_left <= 8 or opp_turns_left <= 8)
        is_early_game = total_turns < 20
        is_mid_game = 20 <= total_turns < 50
        
        # === EGG DIFFERENCE (Primary Objective) ===
        my_eggs = game_board.chicken_player.get_eggs_laid()
        opp_eggs = game_board.chicken_enemy.get_eggs_laid()
        egg_diff = my_eggs - opp_eggs
        
        # Scale egg value by game phase
        if is_endgame:
            egg_weight = self.weights['egg_diff_endgame']
        else:
            egg_weight = self.weights['egg_diff']
        
        score += egg_diff * egg_weight
        
        # CRITICAL: If we're behind and running out of turns, be more aggressive with eggs
        if egg_diff < 0 and my_turns_left < 10:
            score += egg_diff * 50.0  # Extra penalty for being behind in endgame
        
        # === CORNER CONTROL (3x Egg Value) ===
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        my_pos = game_board.chicken_player.get_location()
        opp_pos = game_board.chicken_enemy.get_location()
        
        # Being on corner with ability to lay egg is extremely valuable
        if my_pos in corners:
            score += 10.0  # Bonus just for being on corner
            if game_board.can_lay_egg():
                score += self.weights['corner_egg_bonus']
                # Extra bonus in endgame
                if is_endgame:
                    score += 30.0  # Even more urgent in endgame
        
        # Corner proximity (less important but still valuable)
        my_corner_dist = min(abs(my_pos[0] - c[0]) + abs(my_pos[1] - c[1]) for c in corners)
        opp_corner_dist = min(abs(opp_pos[0] - c[0]) + abs(opp_pos[1] - c[1]) for c in corners)
        score += (opp_corner_dist - my_corner_dist) * self.weights['corner_proximity']
        
        # === MOBILITY ===
        mobility_diff = len(my_moves) - len(opp_moves)
        score += len(my_moves) * self.weights['mobility']
        score += mobility_diff * self.weights['mobility_diff']
        
        # Bonus for being able to lay egg (tactical) - AGGRESSIVE EGG STRATEGY
        if game_board.can_lay_egg():
            score += self.weights['can_lay_egg_bonus']
            # Extra bonus if we're on good parity and not on a trapdoor
            if my_pos not in game_board.found_trapdoors:
                risk_here = self.trap_belief.risk(my_pos)
                if risk_here < 0.1:  # Very safe spot
                    score += 25.0  # Strong bonus for safe egg-laying
                elif risk_here < 0.2:  # Moderately safe
                    score += 15.0  # Good bonus
                elif risk_here < 0.3:  # Acceptable risk
                    score += 8.0
                
                # In endgame, prioritize eggs even more
                if is_endgame:
                    score += 15.0  # Extra egg urgency
        
        # === TRAPDOOR RISK (Avoid Penalties) ===
        # Check if current position is a KNOWN trapdoor (found)
        if my_pos in game_board.found_trapdoors:
            # ABSOLUTELY NEVER go to a known trapdoor!
            score -= 100000.0  # Ultra massive penalty - this should NEVER happen
        else:
            # Use belief-based risk for unknown trapdoors
            risk_at_pos = self.trap_belief.risk(my_pos)
            risk_nearby = self.trap_belief.max_risk_in_radius(my_pos, radius=1)
            
            # Trapdoor penalty is -4 eggs for us, +4 for opponent = 8 egg swing
            # Be ULTRA cautious: triple the risk penalty
            score -= risk_at_pos * self.weights['trapdoor_risk_pos'] * 3.0
            score -= risk_nearby * self.weights['trapdoor_risk_nearby'] * 2.0
            
            # Additional: heavily penalize high-risk squares
            if risk_at_pos > 0.15:
                score -= 400.0  # Strong penalty for risky moves
            if is_endgame and risk_at_pos > 0.10:
                score -= 300.0  # Extra penalty in endgame
        
        # === POSITIONAL CONTROL ===
        center = (3.5, 3.5)
        my_center_dist = abs(my_pos[0] - center[0]) + abs(my_pos[1] - center[1])
        opp_center_dist = abs(opp_pos[0] - center[0]) + abs(opp_pos[1] - center[1])
        
        # Center control more important in midgame
        if is_mid_game:
            center_weight = self.weights['center_control_mid']
        else:
            center_weight = self.weights['center_control_late']
        
        score += (opp_center_dist - my_center_dist) * center_weight
        
        # === TERRITORY/BOARD CONTROL ===
        my_territory = len(game_board.eggs_player) + len(game_board.turds_player)
        opp_territory = len(game_board.eggs_enemy) + len(game_board.turds_enemy)
        territory_diff = my_territory - opp_territory
        score += territory_diff * self.weights['territory_control']
        
        # === TEMPO (Turn Advantage) ===
        turn_diff = my_turns_left - opp_turns_left
        score += turn_diff * self.weights['tempo_advantage']
        
        # === TURD STRATEGY ===
        my_turds_left = game_board.chicken_player.get_turds_left()
        opp_turds_left = game_board.chicken_enemy.get_turds_left()
        turd_diff = my_turds_left - opp_turds_left
        
        # Having turds available is valuable for blocking
        score += turd_diff * self.weights['turd_advantage']
        
        # === CHICKEN DISTANCE (Blocking Strategy) ===
        chicken_distance = abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1])
        
        # Optimal distance is 2 for turd placement
        if chicken_distance == 2:
            score += self.weights['distance_to_opp']
        elif chicken_distance == 1:
            # Too close is slightly bad (can't place turds)
            score -= self.weights['distance_to_opp'] * 0.5
        
        # === TIME MANAGEMENT ===
        my_time = game_board.player_time
        opp_time = game_board.enemy_time
        time_advantage = my_time - opp_time
        
        score += time_advantage * self.weights['time_pressure']
        
        # Penalty if we're critically low on time
        if my_time < 10.0:
            score -= 50.0
        
        return score
    
    def evaluate_move_quality(
        self, 
        game_board: board.Board, 
        move: tuple[enums.Direction, enums.MoveType]
    ) -> float:
        """
        Evaluate the immediate quality of a specific move.
        Used for move ordering in search.
        
        Args:
            game_board: Current board state
            move: Move to evaluate (direction, move_type)
            
        Returns:
            Move quality score (higher is better)
        """
        direction, move_type = move
        score = 0.0
        
        # Prefer egg moves (most important!)
        if move_type == enums.MoveType.EGG:
            score += 100.0  # Increased from 50
            
            # Extra bonus for corner eggs
            my_pos = game_board.chicken_player.get_location()
            corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
            if my_pos in corners:
                score += 150.0  # Increased from 100
        
        # Turd moves are situational
        elif move_type == enums.MoveType.TURD:
            score += 20.0
        
        # Evaluate destination position
        from game.enums import loc_after_direction
        my_pos = game_board.chicken_player.get_location()
        next_pos = loc_after_direction(my_pos, direction)
        
        if game_board.is_valid_cell(next_pos):
            # Prefer moving toward center in early game
            if game_board.turn_count < 20:
                center = (3.5, 3.5)
                center_dist = abs(next_pos[0] - center[0]) + abs(next_pos[1] - center[1])
                score -= center_dist * 2.0
            
            # NEVER move to known trapdoors
            if next_pos in game_board.found_trapdoors:
                score -= 100000.0  # Absolutely forbid this move
            else:
                # Penalize high-risk moves based on belief
                risk = self.trap_belief.risk(next_pos)
                score -= risk * 200.0
            
            # Prefer corner-adjacent in mid/late game
            if game_board.turn_count >= 20:
                corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
                corner_dist = min(abs(next_pos[0] - c[0]) + abs(next_pos[1] - c[1]) for c in corners)
                score -= corner_dist * 3.0
        
        return score

