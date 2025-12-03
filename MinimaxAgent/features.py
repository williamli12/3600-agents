# features.py
"""
Feature extraction from board state for evaluation and learning.
Maps game state + trapdoor beliefs to numeric feature vectors.
"""

from typing import Tuple
import numpy as np
from game import board, enums
from .trapdoor_belief import TrapdoorBelief


def extract_features(game_board: board.Board, trap_belief: TrapdoorBelief) -> np.ndarray:
    """
    Build a comprehensive feature vector describing board state.
    
    Features capture:
    - Material (eggs, turds)
    - Mobility and tempo
    - Positional control
    - Trapdoor risk
    - Game phase indicators
    
    Args:
        game_board: Current board state
        trap_belief: Trapdoor belief tracker
        
    Returns:
        1D float32 numpy array of normalized features
    """
    features = []
    
    # === EGG FEATURES (Primary Objective) ===
    my_eggs = game_board.chicken_player.get_eggs_laid()
    opp_eggs = game_board.chicken_enemy.get_eggs_laid()
    egg_diff = my_eggs - opp_eggs
    egg_sum = my_eggs + opp_eggs
    
    features.extend([
        my_eggs / 30.0,       # Normalize by realistic max
        opp_eggs / 30.0,
        egg_diff / 30.0,
        egg_sum / 60.0,       # Game progress indicator
    ])
    
    # === TURN/TEMPO FEATURES ===
    my_turns_left = game_board.turns_left_player
    opp_turns_left = game_board.turns_left_enemy
    turn_diff = my_turns_left - opp_turns_left
    total_turns = game_board.turn_count
    
    features.extend([
        my_turns_left / 40.0,
        opp_turns_left / 40.0,
        turn_diff / 40.0,
        total_turns / 80.0,  # Game phase (early/mid/late)
    ])
    
    # === MOBILITY FEATURES ===
    my_moves = len(game_board.get_valid_moves())
    
    # Get opponent mobility by reversing perspective
    opp_board = game_board.get_copy()
    opp_board.reverse_perspective()
    opp_moves = len(opp_board.get_valid_moves())
    
    mobility_diff = my_moves - opp_moves
    
    # Count specific move types available
    my_egg_moves = sum(1 for _, mt in game_board.get_valid_moves() if mt == enums.MoveType.EGG)
    my_turd_moves = sum(1 for _, mt in game_board.get_valid_moves() if mt == enums.MoveType.TURD)
    
    features.extend([
        my_moves / 12.0,          # Max ~12 moves
        opp_moves / 12.0,
        mobility_diff / 12.0,
        my_egg_moves / 4.0,       # Max 4 directions
        my_turd_moves / 4.0,
        1.0 if my_moves == 0 else 0.0,    # Blocked indicator
        1.0 if opp_moves == 0 else 0.0,   # Opponent blocked
    ])
    
    # === TRAPDOOR RISK FEATURES ===
    my_pos = game_board.chicken_player.get_location()
    opp_pos = game_board.chicken_enemy.get_location()
    
    risk_at_my_pos = trap_belief.risk(my_pos)
    risk_at_opp_pos = trap_belief.risk(opp_pos)
    max_risk_near_me = trap_belief.max_risk_in_radius(my_pos, radius=1)
    max_risk_near_opp = trap_belief.max_risk_in_radius(opp_pos, radius=1)
    
    features.extend([
        risk_at_my_pos,
        risk_at_opp_pos,
        max_risk_near_me,
        max_risk_near_opp,
    ])
    
    # === SPATIAL CONTROL FEATURES ===
    center = (3.5, 3.5)
    my_center_dist = abs(my_pos[0] - center[0]) + abs(my_pos[1] - center[1])
    opp_center_dist = abs(opp_pos[0] - center[0]) + abs(opp_pos[1] - center[1])
    center_advantage = opp_center_dist - my_center_dist
    
    # Distance between chickens
    chicken_distance = abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1])
    
    # Board control (approximate territory)
    my_territory = len(game_board.eggs_player) + len(game_board.turds_player)
    opp_territory = len(game_board.eggs_enemy) + len(game_board.turds_enemy)
    
    features.extend([
        my_center_dist / 7.0,
        opp_center_dist / 7.0,
        center_advantage / 7.0,
        chicken_distance / 14.0,
        my_territory / 45.0,      # Max ~40 eggs + 5 turds
        opp_territory / 45.0,
    ])
    
    # === TURD/RESOURCE FEATURES ===
    my_turds_left = game_board.chicken_player.get_turds_left()
    opp_turds_left = game_board.chicken_enemy.get_turds_left()
    my_turds_placed = 5 - my_turds_left
    opp_turds_placed = 5 - opp_turds_left
    
    features.extend([
        my_turds_left / 5.0,
        opp_turds_left / 5.0,
        my_turds_placed / 5.0,
        opp_turds_placed / 5.0,
        (my_turds_left - opp_turds_left) / 5.0,
    ])
    
    # === CORNER CONTROL FEATURES ===
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    my_on_corner = 1.0 if my_pos in corners else 0.0
    opp_on_corner = 1.0 if opp_pos in corners else 0.0
    
    # Distance to nearest corner
    my_corner_dist = min(abs(my_pos[0] - c[0]) + abs(my_pos[1] - c[1]) for c in corners)
    opp_corner_dist = min(abs(opp_pos[0] - c[0]) + abs(opp_pos[1] - c[1]) for c in corners)
    
    # Can we lay an egg right now (important tactical info)
    can_lay_egg = 1.0 if game_board.can_lay_egg() else 0.0
    can_lay_egg_on_corner = 1.0 if (my_on_corner and can_lay_egg) else 0.0
    
    features.extend([
        my_on_corner,
        opp_on_corner,
        my_corner_dist / 14.0,
        opp_corner_dist / 14.0,
        can_lay_egg,
        can_lay_egg_on_corner,
    ])
    
    # === PARITY FEATURES (Egg-laying constraints) ===
    my_parity = (my_pos[0] + my_pos[1]) % 2
    opp_parity = (opp_pos[0] + opp_pos[1]) % 2
    can_lay_at_pos = 1.0 if my_parity == game_board.chicken_player.even_chicken else 0.0
    
    features.extend([
        float(my_parity),
        float(opp_parity),
        can_lay_at_pos,
    ])
    
    # === TIME/RESOURCE PRESSURE ===
    my_time = game_board.player_time
    opp_time = game_board.enemy_time
    time_pressure = 1.0 if my_time < 30.0 else 0.0  # Under 30 seconds is critical
    
    features.extend([
        np.clip(my_time / 360.0, 0, 1),
        np.clip(opp_time / 360.0, 0, 1),
        (my_time - opp_time) / 360.0,
        time_pressure,
    ])
    
    # === GAME PHASE INDICATORS ===
    early_game = 1.0 if total_turns < 20 else 0.0
    mid_game = 1.0 if 20 <= total_turns < 50 else 0.0
    late_game = 1.0 if total_turns >= 50 else 0.0
    endgame = 1.0 if (my_turns_left <= 5 or opp_turns_left <= 5) else 0.0
    
    features.extend([
        early_game,
        mid_game,
        late_game,
        endgame,
    ])
    
    # === TERMINAL STATE INDICATORS ===
    is_terminal = 1.0 if game_board.is_game_over() else 0.0
    
    features.append(is_terminal)
    
    return np.array(features, dtype=np.float32)


def get_feature_dim() -> int:
    """
    Return the dimensionality of the feature vector.
    
    Returns:
        Number of features
    """
    # Count: 4 (eggs) + 4 (turns) + 7 (mobility) + 4 (risk) + 6 (spatial) +
    #        5 (turds) + 6 (corners) + 3 (parity) + 4 (time) + 4 (phase) + 1 (terminal)
    return 48


def quick_tactical_score(game_board: board.Board) -> float:
    """
    Ultra-fast tactical evaluation for move ordering.
    Only considers immediate material and critical threats.
    
    Args:
        game_board: Board state
        
    Returns:
        Quick score estimate
    """
    score = 0.0
    
    # Egg difference (primary)
    egg_diff = game_board.chicken_player.get_eggs_laid() - game_board.chicken_enemy.get_eggs_laid()
    score += egg_diff * 100.0
    
    # Mobility
    my_moves = len(game_board.get_valid_moves())
    score += my_moves * 3.0
    
    if my_moves == 0:
        score -= 1000.0  # Critical: blocked
    
    # Corner bonus
    my_pos = game_board.chicken_player.get_location()
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    if my_pos in corners and game_board.can_lay_egg():
        score += 30.0
    
    return score

