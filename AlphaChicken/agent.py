"""
agent.py - Main agent interface for AlphaChicken

This agent uses a hybrid approach:
1. Use the trained Neural Network (Policy Head) to rank moves by "expert intuition".
2. Filter out obviously bad moves (invalid, walking into known trapdoors).
3. (Optional future) Search top N moves. Currently just takes the best safe move.
"""

import os
import time
from collections.abc import Callable
from typing import List, Tuple, Optional
import numpy as np
import torch

from game import board, enums

try:
    from .model import AlphaChickenNet, board_to_tensor, index_to_move, move_to_index
except ImportError:
    from model import AlphaChickenNet, board_to_tensor, index_to_move, move_to_index


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
    AlphaChicken agent using deep RL Policy Network directly.
    Uses the network to propose moves, filters them for safety, and picks the best one.
    """
    
    def __init__(self, board: board.Board, time_left: Callable, seed: Optional[int] = None):
        """
        Initialize the agent.
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
        
        self.turn_count = 0
    
    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Choose a move using Policy Network + Safety Checks.
        """
        self.turn_count += 1
        
        # Update trapdoor beliefs
        self.trap_belief.update(board, sensor_data)
        
        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return (enums.Direction.UP, enums.MoveType.PLAIN) # Should not happen
            
        # 1. Safety Filter: Don't walk into known trapdoors
        safe_moves = []
        my_pos = board.chicken_player.get_location()
        
        for m in valid_moves:
            # Predict next location
            next_pos = board.chicken_player.get_next_loc(m[0], my_pos)
            
            # Check if it's a known trapdoor
            if next_pos in board.found_trapdoors:
                continue
                
            # Check probabilistic trapdoor risk
            # Get risk from belief maps
            if next_pos is not None:
                x, y = next_pos
                parity = (x + y) % 2
                if parity == 0:
                    risk = self.trap_belief.belief_even[y, x]
                else:
                    risk = self.trap_belief.belief_odd[y, x]
                
                # If risk is too high (> 50%), avoid unless desperate
                if risk > 0.50:
                    continue
            
            safe_moves.append(m)
            
        # If no safe moves, revert to all valid moves
        moves_to_consider = safe_moves if safe_moves else valid_moves
        
        # 2. Neural Network Policy
        # Prepare input tensor
        state_tensor = board_to_tensor(board, self.trap_belief)
        state_tensor = state_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
        # 3. Pick Best Move from Candidates
        best_move = None
        best_score = -1.0
        
        for move in moves_to_consider:
            # Convert move to index
            idx = move_to_index(move[0], move[1])
            
            # Score is the network's probability for this move
            score = policy_probs[idx]
            
            # Bonus: Prefer Egg moves slightly if score is close
            if move[1] == enums.MoveType.EGG:
                score *= 1.1
            
            if score > best_score:
                best_score = score
                best_move = move
                
        # Fallback
        if best_move is None:
            best_move = valid_moves[0]
            
        return best_move
