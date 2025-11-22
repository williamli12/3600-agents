"""
model.py - Neural Network Architecture for AlphaChicken

Implements a convolutional residual network with policy and value heads.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from game import board, enums


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaChickenNet(nn.Module):
    """
    AlphaZero-style network for Chicken game.
    
    Input: 8x8x14 board representation
    Output: 
        - Policy: 12-dimensional vector (4 directions × 3 move types)
        - Value: Scalar in [-1, 1] (expected game outcome)
    """
    
    def __init__(self, num_channels: int = 128, num_res_blocks: int = 4):
        super().__init__()
        
        # Input block
        self.input_conv = nn.Conv2d(14, num_channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 12)  # 4 directions × 3 move types
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch, 14, 8, 8)
            
        Returns:
            policy_logits: Tensor of shape (batch, 12) - log probabilities
            value: Tensor of shape (batch, 1) - expected outcome in [-1, 1]
        """
        # Input block
        out = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual tower
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value


def board_to_tensor(game_board: board.Board, trap_belief=None) -> torch.Tensor:
    """
    Convert board state to neural network input tensor.
    
    Channel layout (14 channels total):
    0: My chicken position (one-hot)
    1: Enemy chicken position (one-hot)
    2: My eggs (binary map)
    3: Enemy eggs (binary map)
    4: My turds (binary map)
    5: Enemy turds (binary map)
    6-7: Trapdoor belief maps (probability grids)
    8: Can lay egg here? (binary map based on parity)
    9: Turns left (normalized, filled plane)
    10: My turds left (normalized, filled plane)
    11: Enemy turds left (normalized, filled plane)
    12: Corner locations (static binary map)
    13: Distance to center (normalized static map)
    
    Args:
        game_board: Current Board object
        trap_belief: Optional TrapdoorBelief object for channels 6-7
        
    Returns:
        Tensor of shape (1, 14, 8, 8)
    """
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Channel 0: My chicken position
    my_pos = game_board.chicken_player.get_location()
    tensor[0, my_pos[1], my_pos[0]] = 1.0
    
    # Channel 1: Enemy chicken position
    enemy_pos = game_board.chicken_enemy.get_location()
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
    
    # Channels 6-7: Trapdoor belief maps
    if trap_belief is not None:
        # Assuming trap_belief has .belief_even and .belief_odd as 8x8 arrays
        if hasattr(trap_belief, 'belief_even'):
            tensor[6] = trap_belief.belief_even
        if hasattr(trap_belief, 'belief_odd'):
            tensor[7] = trap_belief.belief_odd
    
    # Channel 8: Can lay egg at each position (parity check)
    even_chicken = game_board.chicken_player.even_chicken
    for y in range(8):
        for x in range(8):
            if (x + y) % 2 == even_chicken:
                tensor[8, y, x] = 1.0
    
    # Channel 9: Turns left (normalized)
    max_turns = getattr(game_board, 'MAX_TURNS', 40.0)
    turns_left_norm = game_board.turns_left_player / max_turns
    tensor[9, :, :] = turns_left_norm
    
    # Channel 10: My turds left (normalized)
    my_turds_norm = game_board.chicken_player.get_turds_left() / 5.0
    tensor[10, :, :] = my_turds_norm
    
    # Channel 11: Enemy turds left (normalized)
    enemy_turds_norm = game_board.chicken_enemy.get_turds_left() / 5.0
    tensor[11, :, :] = enemy_turds_norm
    
    # Channel 12: Corner locations
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for corner in corners:
        tensor[12, corner[1], corner[0]] = 1.0
    
    # Channel 13: Distance to center (normalized)
    center = 3.5
    for y in range(8):
        for x in range(8):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            tensor[13, y, x] = 1.0 - (dist / (center * np.sqrt(2)))  # Normalize to [0, 1]
    
    # Convert to torch tensor and add batch dimension
    return torch.from_numpy(tensor).unsqueeze(0)


def index_to_move(index: int) -> Tuple[enums.Direction, enums.MoveType]:
    """
    Convert network output index (0-11) to (Direction, MoveType).
    
    Mapping:
    0-2: UP (PLAIN, EGG, TURD)
    3-5: RIGHT (PLAIN, EGG, TURD)
    6-8: DOWN (PLAIN, EGG, TURD)
    9-11: LEFT (PLAIN, EGG, TURD)
    """
    direction = enums.Direction(index // 3)
    move_type = enums.MoveType(index % 3)
    return (direction, move_type)


def move_to_index(direction: enums.Direction, move_type: enums.MoveType) -> int:
    """Convert (Direction, MoveType) to network output index (0-11)."""
    return int(direction) * 3 + int(move_type)


def get_valid_move_mask(game_board: board.Board) -> np.ndarray:
    """
    Get binary mask for valid moves.
    
    Returns:
        Array of shape (12,) where 1 = valid, 0 = invalid
    """
    mask = np.zeros(12, dtype=np.float32)
    valid_moves = game_board.get_valid_moves()
    
    for direction, move_type in valid_moves:
        index = move_to_index(direction, move_type)
        mask[index] = 1.0
    
    return mask

