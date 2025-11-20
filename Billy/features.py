# features.py
"""
Feature extraction from board state for evaluation and learning.
Maps game state + trapdoor beliefs to a numeric feature vector.
"""

from typing import Tuple
import numpy as np
from game import board
from .trapdoor_belief import TrapdoorBelief


def extract_features(game_board: board.Board, trap_belief: TrapdoorBelief) -> np.ndarray:
    """
    Build a feature vector describing board state from the current player's perspective.
    
    Features include:
    - Egg counts and difference
    - Turn counts and tempo
    - Mobility (valid moves)
    - Trapdoor risk
    - Spatial control
    - Turds remaining
    
    Args:
        game_board: Current board state
        trap_belief: Trapdoor belief tracker
        
    Returns:
        1D float32 numpy array of features
    """
    features = []
    
    # === EGG FEATURES ===
    my_eggs = game_board.chicken_player.get_eggs_laid()
    opp_eggs = game_board.chicken_enemy.get_eggs_laid()
    egg_diff = my_eggs - opp_eggs
    
    features.extend([
        my_eggs / 20.0,      # Normalize by reasonable max
        opp_eggs / 20.0,
        egg_diff / 20.0,
    ])
    
    # === TURN/TEMPO FEATURES ===
    my_turns_left = game_board.turns_left_player
    opp_turns_left = game_board.turns_left_enemy
    turn_diff = my_turns_left - opp_turns_left
    
    features.extend([
        my_turns_left / 40.0,
        opp_turns_left / 40.0,
        turn_diff / 40.0,
    ])
    
    # === MOBILITY FEATURES ===
    my_moves = len(game_board.get_valid_moves())
    
    # Estimate opponent moves by reversing perspective
    opp_board = game_board.get_copy()
    opp_board.reverse_perspective()
    opp_moves = len(opp_board.get_valid_moves())
    
    mobility_diff = my_moves - opp_moves
    
    features.extend([
        my_moves / 12.0,     # Max ~12 moves (4 directions * 3 move types)
        opp_moves / 12.0,
        mobility_diff / 12.0,
        1.0 if my_moves == 0 else 0.0,  # Critical: blocked indicator
        1.0 if opp_moves == 0 else 0.0,  # Opponent blocked
    ])
    
    # === TRAPDOOR RISK FEATURES ===
    my_pos = game_board.chicken_player.get_location()
    risk_here = trap_belief.risk(my_pos)
    max_risk_nearby = trap_belief.max_risk_in_radius(my_pos, radius=1)
    
    features.extend([
        risk_here,
        max_risk_nearby,
    ])
    
    # === SPATIAL CONTROL FEATURES ===
    # Distance from center (board center is at 3.5, 3.5)
    center = (3.5, 3.5)
    my_center_dist = abs(my_pos[0] - center[0]) + abs(my_pos[1] - center[1])
    
    opp_pos = game_board.chicken_enemy.get_location()
    opp_center_dist = abs(opp_pos[0] - center[0]) + abs(opp_pos[1] - center[1])
    
    center_advantage = opp_center_dist - my_center_dist
    
    # Distance between chickens
    chicken_distance = abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1])
    
    features.extend([
        my_center_dist / 7.0,      # Max Manhattan distance from center
        opp_center_dist / 7.0,
        center_advantage / 7.0,
        chicken_distance / 14.0,   # Max distance on 8x8 board
    ])
    
    # === TURD FEATURES ===
    my_turds_left = game_board.chicken_player.get_turds_left()
    opp_turds_left = game_board.chicken_enemy.get_turds_left()
    my_turds_placed = game_board.chicken_player.get_turds_placed()
    opp_turds_placed = game_board.chicken_enemy.get_turds_placed()
    
    features.extend([
        my_turds_left / 5.0,
        opp_turds_left / 5.0,
        my_turds_placed / 5.0,
        opp_turds_placed / 5.0,
    ])
    
    # === CORNER CONTROL FEATURES ===
    # Check if we're on or near a corner (corners give 3x eggs)
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    my_on_corner = 1.0 if my_pos in corners else 0.0
    opp_on_corner = 1.0 if opp_pos in corners else 0.0
    
    # Distance to nearest corner
    my_corner_dist = min(abs(my_pos[0] - c[0]) + abs(my_pos[1] - c[1]) for c in corners)
    opp_corner_dist = min(abs(opp_pos[0] - c[0]) + abs(opp_pos[1] - c[1]) for c in corners)
    
    features.extend([
        my_on_corner,
        opp_on_corner,
        my_corner_dist / 14.0,
        opp_corner_dist / 14.0,
    ])
    
    # === GAME STATE FEATURES ===
    # Time remaining
    my_time = game_board.player_time
    opp_time = game_board.enemy_time
    
    features.extend([
        my_time / 360.0,  # Normalize by 6 minutes
        opp_time / 360.0,
        (my_time - opp_time) / 360.0,
    ])
    
    # Game over indicator
    features.append(1.0 if game_board.is_game_over() else 0.0)
    
    return np.array(features, dtype=np.float32)


def get_feature_dim() -> int:
    """
    Return the dimensionality of the feature vector.
    
    Returns:
        Number of features
    """
    # Count: 3 (eggs) + 3 (turns) + 5 (mobility) + 2 (risk) + 
    #        4 (spatial) + 4 (turds) + 4 (corners) + 3 (time) + 1 (game over)
    return 29

