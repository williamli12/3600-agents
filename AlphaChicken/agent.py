"""
agent.py - Main agent interface for AlphaChicken

This is the entry point called by the game engine.
"""

import os
import time
from collections.abc import Callable
from typing import List, Tuple, Optional
import numpy as np
import torch

from game import board, enums

try:
    from .model import AlphaChickenNet
    from .mcts import MCTS
except ImportError:
    from model import AlphaChickenNet
    from mcts import MCTS


class TrapdoorBelief:
    """
    Lightweight Bayesian belief tracker for trapdoors.
    Maintains probability distributions over trapdoor locations.
    """
    
    def __init__(self, board_size: int = 8):
        self.size = board_size
        # Separate belief for even and odd parity trapdoors
        # Initialize with center-weighted distribution per game rules
        self.belief_even = np.zeros((board_size, board_size), dtype=np.float32)
        self.belief_odd = np.zeros((board_size, board_size), dtype=np.float32)
        
        for x in range(self.size):
            for y in range(self.size):
                # Distance from nearest edge
                dist_from_edge = min(x, y, self.size - 1 - x, self.size - 1 - y)
                
                # Weights from rules: Edge=0, Layer1=0, Layer2=1, Center=2
                if dist_from_edge <= 1:
                    weight = 0.001  # Small epsilon, not 0
                elif dist_from_edge == 2:
                    weight = 1.0
                else:
                    weight = 2.0
                
                if (x + y) % 2 == 0:
                    self.belief_even[y, x] = weight
                else:
                    self.belief_odd[y, x] = weight
        
        self._normalize()
    
    def _normalize(self):
        """Normalize beliefs to sum to 1."""
        self.belief_even = self.belief_even / self.belief_even.sum()
        self.belief_odd = self.belief_odd / self.belief_odd.sum()
    
    def update(self, game_board: board.Board, sensor_data: List[Tuple[bool, bool]]):
        """
        Update beliefs based on sensor observations.
        
        Args:
            game_board: Current board state
            sensor_data: [(heard, felt) for trap_0, (heard, felt) for trap_1]
        """
        player_pos = game_board.chicken_player.get_location()
        
        # Update for each trapdoor
        for trap_idx, (heard, felt) in enumerate(sensor_data):
            # Determine which belief grid to update based on parity
            if trap_idx == 0:
                belief = self.belief_even
            else:
                belief = self.belief_odd
            
            # Bayesian update for each position
            for y in range(self.size):
                for x in range(self.size):
                    # Skip found trapdoors
                    if (x, y) in game_board.found_trapdoors:
                        belief[y, x] = 0.0
                        continue
                    
                    # Calculate distance
                    dx = abs(x - player_pos[0])
                    dy = abs(y - player_pos[1])
                    
                    # Likelihood of observation given trapdoor at (x, y)
                    likelihood = self._calc_likelihood(dx, dy, heard, felt)
                    belief[y, x] *= likelihood
            
            # Store back
            if trap_idx == 0:
                self.belief_even = belief
            else:
                self.belief_odd = belief
        
        self._normalize()
    
    def _calc_likelihood(self, dx: int, dy: int, heard: bool, felt: bool) -> float:
        """Calculate P(observation | trapdoor at distance)."""
        # Special case: standing on the trapdoor
        if dx == 0 and dy == 0:
            return 1.0
        
        # Probability of hearing
        if dx > 2 or dy > 2 or (dx == 2 and dy == 2):
            p_hear = 0.0
        elif dx == 2 or dy == 2:
            p_hear = 0.1
        elif dx == 1 and dy == 1:
            p_hear = 0.25
        elif dx == 1 or dy == 1:
            p_hear = 0.5
        else:
            p_hear = 0.0
        
        # Probability of feeling
        if dx > 1 or dy > 1:
            p_feel = 0.0
        elif dx == 1 and dy == 1:
            p_feel = 0.15
        elif dx == 1 or dy == 1:
            p_feel = 0.3
        else:
            p_feel = 0.0
        
        # Likelihood of observation
        likelihood = 1.0
        if heard:
            likelihood *= p_hear if p_hear > 0 else 0.01  # Small epsilon
        else:
            likelihood *= (1 - p_hear) if p_hear < 1 else 0.01
        
        if felt:
            likelihood *= p_feel if p_feel > 0 else 0.01
        else:
            likelihood *= (1 - p_feel) if p_feel < 1 else 0.01
        
        return likelihood


class PlayerAgent:
    """
    AlphaChicken agent using deep RL and MCTS.
    """
    
    def __init__(self, board: board.Board, time_left: Callable, seed: Optional[int] = None):
        """
        Initialize the agent.
        
        Args:
            board: Initial game board
            time_left: Function returning remaining time in seconds
            seed: Random seed (optional)
        """
        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize neural network
        self.model = AlphaChickenNet(num_channels=128, num_res_blocks=4)
        self.model.to(self.device)
        
        # Try to load trained weights
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[AlphaChicken] Loaded model from {model_path}")
            except Exception as e:
                print(f"[AlphaChicken] Warning: Could not load model: {e}")
                print("[AlphaChicken] Using randomly initialized network")
        else:
            print(f"[AlphaChicken] Warning: {model_path} not found")
            print("[AlphaChicken] Using randomly initialized network")
        
        self.model.eval()
        
        # Initialize trapdoor belief tracker
        self.trap_belief = TrapdoorBelief(board_size=8)
        
        # Initialize MCTS
        self.mcts = MCTS(self.model, self.trap_belief, c_puct=1.5, device=self.device)
        
        # Time management
        self.time_safety_margin = 2.0  # Reserve 2 seconds
        self.turn_count = 0
    
    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Choose a move using MCTS.
        
        Args:
            board: Current board state
            sensor_data: Sensor readings [(heard, felt) for each trapdoor]
            time_left: Function returning remaining time
            
        Returns:
            (Direction, MoveType) tuple
        """
        self.turn_count += 1
        
        # Update trapdoor beliefs
        self.trap_belief.update(board, sensor_data)
        
        # Calculate time budget
        remaining_time = time_left()
        turns_remaining = max(board.turns_left_player, 1)
        
        # Allocate time: use a fraction of remaining time
        time_per_move = (remaining_time - self.time_safety_margin) / turns_remaining
        time_budget = max(min(time_per_move * 0.9, 8.0), 0.5)  # Between 0.5 and 8 seconds (increased thinking time)
        
        # Run MCTS
        start_time = time.time()
        simulations = 0
        max_simulations = 1000
        
        root = None
        while time.time() - start_time < time_budget and simulations < max_simulations:
            # Run one batch of simulations (larger batches for better GPU utilization)
            batch_size = 25
            root = self.mcts.search(board, num_simulations=batch_size)
            simulations += batch_size
            
            # Early exit if very confident
            if simulations >= 50:
                best_move = self.mcts.get_best_move(root)
                best_child = root.children.get(best_move)
                if best_child and best_child.visit_count > simulations * 0.8:
                    break  # One move dominates
        
        # Get best move
        if root is None:
            # Fallback: no time for search
            valid_moves = board.get_valid_moves()
            if valid_moves:
                return valid_moves[0]
            return (enums.Direction.UP, enums.MoveType.PLAIN)
        
        move = self.mcts.get_best_move(root)
        
        # Validate move
        valid_moves = board.get_valid_moves()
        if move not in valid_moves:
            # Fallback to valid move
            if valid_moves:
                move = valid_moves[0]
            else:
                move = (enums.Direction.UP, enums.MoveType.PLAIN)
        
        elapsed = time.time() - start_time
        print(f"[AlphaChicken] Turn {self.turn_count}: {simulations} sims in {elapsed:.2f}s, move={move}")
        
        return move

