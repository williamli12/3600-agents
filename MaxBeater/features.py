"""
features.py - Feature extraction for evaluation and value model

Provides two types of features:
1. Scalar features - hand-crafted features for heuristic
2. Board tensor - spatial representation for neural network
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from game import board
from .belief import TrapdoorBelief, Coord


def extract_scalar_features(game_board: board.Board, trap_belief: TrapdoorBelief) -> np.ndarray:
    """
    Extract hand-crafted scalar features for evaluation.
    
    Features include:
    - Egg differential (primary objective)
    - Mobility (# valid moves for us vs enemy)
    - Turns remaining
    - Turd counts
    - Positional features (distance to center, corners)
    - Trapdoor risk at current and nearby positions
    
    Returns:
        1D float32 array of features
    """
    features = []
    
    # === Game state features ===
    my_eggs = game_board.chicken_player.get_eggs_laid()
    enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
    features.append(float(my_eggs - enemy_eggs))  # Egg differential
    features.append(float(my_eggs))
    features.append(float(enemy_eggs))
    
    # === Mobility features ===
    my_moves = len(game_board.get_valid_moves(enemy=False))
    features.append(float(my_moves))
    
    # Enemy mobility (need to check from enemy perspective)
    enemy_board = game_board.get_copy()
    enemy_board.reverse_perspective()
    enemy_moves = len(enemy_board.get_valid_moves(enemy=False))
    features.append(float(enemy_moves))
    features.append(float(my_moves - enemy_moves))  # Mobility differential
    
    # === Turn and time features ===
    turns_left_me = game_board.turns_left_player
    turns_left_enemy = game_board.turns_left_enemy
    features.append(float(turns_left_me))
    features.append(float(turns_left_enemy))
    features.append(float(game_board.turn_count))
    
    # Normalized turn progress (0 = start, 1 = end)
    max_turns = game_board.MAX_TURNS
    turn_progress = game_board.turn_count / (2.0 * max_turns) if max_turns > 0 else 0.0
    features.append(turn_progress)
    
    # === Turd features ===
    my_turds = game_board.chicken_player.get_turds_left()
    enemy_turds = game_board.chicken_enemy.get_turds_left()
    features.append(float(my_turds))
    features.append(float(enemy_turds))
    features.append(float(my_turds - enemy_turds))
    
    # === Positional features ===
    my_pos = game_board.chicken_player.get_location()
    enemy_pos = game_board.chicken_enemy.get_location()
    
    # Distance to center (normalized)
    center = game_board.game_map.MAP_SIZE / 2.0
    my_dist_center = (abs(my_pos[0] - center) + abs(my_pos[1] - center)) / game_board.game_map.MAP_SIZE
    enemy_dist_center = (abs(enemy_pos[0] - center) + abs(enemy_pos[1] - center)) / game_board.game_map.MAP_SIZE
    features.append(my_dist_center)
    features.append(enemy_dist_center)
    
    # Distance between chickens
    chicken_dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
    features.append(float(chicken_dist) / game_board.game_map.MAP_SIZE)
    
    # Corner proximity (corners are valuable: 3 eggs instead of 1)
    corners = [
        (0, 0),
        (0, game_board.game_map.MAP_SIZE - 1),
        (game_board.game_map.MAP_SIZE - 1, 0),
        (game_board.game_map.MAP_SIZE - 1, game_board.game_map.MAP_SIZE - 1)
    ]
    my_corner_dist = min(abs(my_pos[0] - c[0]) + abs(my_pos[1] - c[1]) for c in corners)
    enemy_corner_dist = min(abs(enemy_pos[0] - c[0]) + abs(enemy_pos[1] - c[1]) for c in corners)
    features.append(float(my_corner_dist) / game_board.game_map.MAP_SIZE)
    features.append(float(enemy_corner_dist) / game_board.game_map.MAP_SIZE)
    
    # Am I on a corner?
    my_on_corner = 1.0 if my_pos in corners else 0.0
    enemy_on_corner = 1.0 if enemy_pos in corners else 0.0
    features.append(my_on_corner)
    features.append(enemy_on_corner)
    
    # Can I lay an egg right now?
    can_lay_egg = 1.0 if game_board.can_lay_egg() else 0.0
    features.append(can_lay_egg)
    
    # === Trapdoor risk features ===
    my_trap_risk = trap_belief.risk(my_pos)
    enemy_trap_risk = trap_belief.risk(enemy_pos)
    features.append(my_trap_risk)
    features.append(enemy_trap_risk)
    
    # Max risk in radius
    my_max_risk_nearby = trap_belief.max_risk_in_radius(my_pos, radius=2)
    features.append(my_max_risk_nearby)
    
    return np.array(features, dtype=np.float32)


def encode_board_tensor(game_board: board.Board, trap_belief: TrapdoorBelief) -> np.ndarray:
    """
    Encode board state as a 3D tensor for neural network input.
    
    Format: (C, H, W) where C = channels, H = W = 8
    
    Channels:
    0: My chicken position (one-hot)
    1: Enemy chicken position (one-hot)
    2: My eggs (binary)
    3: Enemy eggs (binary)
    4: My turds (binary)
    5: Enemy turds (binary)
    6: Trapdoor belief (combined probability)
    7: Found trapdoors (binary)
    8: Valid egg positions for me (parity)
    9: Valid egg positions for enemy (parity)
    10: Corners (static binary)
    11: Distance to center (gradient)
    12: Turn progress (broadcast scalar)
    13: Egg differential (broadcast scalar)
    
    Returns:
        3D float32 array of shape (14, 8, 8)
    """
    size = game_board.game_map.MAP_SIZE
    num_channels = 14
    tensor = np.zeros((num_channels, size, size), dtype=np.float32)
    
    my_pos = game_board.chicken_player.get_location()
    enemy_pos = game_board.chicken_enemy.get_location()
    
    # Channel 0: My chicken position (one-hot)
    tensor[0, my_pos[1], my_pos[0]] = 1.0
    
    # Channel 1: Enemy chicken position (one-hot)
    tensor[1, enemy_pos[1], enemy_pos[0]] = 1.0
    
    # Channel 2: My eggs
    for egg_pos in game_board.eggs_player:
        tensor[2, egg_pos[1], egg_pos[0]] = 1.0
    
    # Channel 3: Enemy eggs
    for egg_pos in game_board.eggs_enemy:
        tensor[3, egg_pos[1], egg_pos[0]] = 1.0
    
    # Channel 4: My turds
    for turd_pos in game_board.turds_player:
        tensor[4, turd_pos[1], turd_pos[0]] = 1.0
    
    # Channel 5: Enemy turds
    for turd_pos in game_board.turds_enemy:
        tensor[5, turd_pos[1], turd_pos[0]] = 1.0
    
    # Channel 6: Trapdoor belief (probability grid)
    tensor[6, :, :] = trap_belief.get_belief_grid()
    
    # Channel 7: Found trapdoors
    for trap_pos in game_board.found_trapdoors:
        tensor[7, trap_pos[1], trap_pos[0]] = 1.0
    
    # Channel 8: Valid egg positions for me (parity)
    my_parity = game_board.chicken_player.even_chicken
    for y in range(size):
        for x in range(size):
            if (x + y) % 2 == my_parity:
                tensor[8, y, x] = 1.0
    
    # Channel 9: Valid egg positions for enemy (parity)
    enemy_parity = game_board.chicken_enemy.even_chicken
    for y in range(size):
        for x in range(size):
            if (x + y) % 2 == enemy_parity:
                tensor[9, y, x] = 1.0
    
    # Channel 10: Corners (static)
    tensor[10, 0, 0] = 1.0
    tensor[10, 0, size-1] = 1.0
    tensor[10, size-1, 0] = 1.0
    tensor[10, size-1, size-1] = 1.0
    
    # Channel 11: Distance to center (gradient)
    center = size / 2.0
    for y in range(size):
        for x in range(size):
            dist = (abs(x - center) + abs(y - center)) / size
            tensor[11, y, x] = 1.0 - dist  # Inverted: center = 1, edges = 0
    
    # Channel 12: Turn progress (broadcast)
    turn_progress = game_board.turn_count / (2.0 * game_board.MAX_TURNS)
    tensor[12, :, :] = turn_progress
    
    # Channel 13: Egg differential (broadcast, normalized)
    egg_diff = game_board.chicken_player.get_eggs_laid() - game_board.chicken_enemy.get_eggs_laid()
    # Normalize to roughly [-1, 1] assuming max ~20 egg difference
    tensor[13, :, :] = np.clip(egg_diff / 20.0, -1.0, 1.0)
    
    return tensor


def get_feature_dimensions() -> Tuple[int, Tuple[int, int, int]]:
    """
    Return the dimensions of scalar features and board tensor.
    
    Returns:
        (num_scalar_features, (channels, height, width))
    """
    return (24, (14, 8, 8))  # 24 scalar features, 14x8x8 tensor

