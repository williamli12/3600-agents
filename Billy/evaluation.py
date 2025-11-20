# evaluation.py
"""
Board evaluation combining hand-crafted heuristics with optional learned value network.
"""

from __future__ import annotations
from typing import Optional
import os
import numpy as np
from game import board
from .trapdoor_belief import TrapdoorBelief
from .features import extract_features, get_feature_dim

# Try to import PyTorch for optional value network
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


class ValueNet(nn.Module):
    """
    Tiny MLP that maps feature vectors to a scalar value in [-1, 1].
    Only used if torch is available and value_net.pt exists.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize the value network.
        
        Args:
            input_dim: Size of input feature vector
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)


class Evaluator:
    """
    Evaluates board positions using heuristics and optionally a learned value function.
    """
    
    def __init__(self, trap_belief: TrapdoorBelief):
        """
        Initialize the evaluator.
        
        Args:
            trap_belief: Trapdoor belief tracker for risk assessment
        """
        self.trap_belief = trap_belief
        self.use_value_net = False
        self.value_net: Optional[ValueNet] = None
        self.input_dim: Optional[int] = None
        self._maybe_load_value_net()
    
    def _maybe_load_value_net(self) -> None:
        """
        If value_net.pt exists and torch is available, initialize ValueNet,
        infer input_dim from feature extractor, load weights, and enable it.
        """
        if not TORCH_AVAILABLE:
            return
        
        # Look for value_net.pt in the same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'value_net.pt')
        
        if not os.path.exists(model_path):
            return
        
        try:
            # Get feature dimension
            self.input_dim = get_feature_dim()
            
            # Create and load model
            self.value_net = ValueNet(self.input_dim, hidden_dim=64)
            self.value_net.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.value_net.eval()
            self.use_value_net = True
        except Exception as e:
            # If loading fails, just use heuristic
            self.use_value_net = False
            self.value_net = None
    
    def heuristic(self, game_board: board.Board) -> float:
        """
        Hand-crafted heuristic evaluation function.
        
        PERSPECTIVE: This function ALWAYS evaluates from the perspective of
        board.chicken_player. The search engine ensures that chicken_player is
        consistently set to our agent in both max and min nodes, so this evaluation
        naturally works correctly in adversarial search.
        
        Combines multiple strategic factors with carefully tuned weights:
        - Egg difference (primary objective, with endgame scaling)
        - Mobility (avoid getting blocked)
        - Trapdoor risk (avoid penalties)
        - Positional control (center, corners)
        - Turn/tempo advantage
        
        Args:
            game_board: Board state to evaluate
            
        Returns:
            Heuristic score from chicken_player's perspective (higher is better)
        """
        score = 0.0
        
        # === TERMINAL/CRITICAL STATES ===
        if game_board.is_game_over():
            winner = game_board.get_winner()
            from game.enums import Result
            if winner == Result.PLAYER:
                return 10000.0  # We won
            elif winner == Result.ENEMY:
                return -10000.0  # We lost
            else:
                return 0.0  # Tie
        
        # Check if we're blocked (no valid moves)
        my_moves = game_board.get_valid_moves()
        if len(my_moves) == 0:
            # Being blocked is catastrophic (-5 eggs penalty + positional loss)
            # Reduced from 5000 to 3000 to not completely overshadow large egg leads
            score -= 3000.0
        
        # Check if opponent is blocked
        opp_board = game_board.get_copy()
        opp_board.reverse_perspective()
        opp_moves = opp_board.get_valid_moves()
        if len(opp_moves) == 0:
            # Blocking opponent is great (+5 eggs gain)
            # Reduced from 5000 to 3000 for better balance
            score += 3000.0
        
        # === EGG DIFFERENCE (PRIMARY OBJECTIVE) ===
        my_eggs = game_board.chicken_player.get_eggs_laid()
        opp_eggs = game_board.chicken_enemy.get_eggs_laid()
        egg_diff = my_eggs - opp_eggs
        
        # Add endgame awareness: egg difference becomes more important as game progresses
        # In the final 10 turns, we increase weight on raw egg count
        min_turns_left = min(game_board.turns_left_player, game_board.turns_left_enemy)
        endgame_factor = 1.0 + max(0, 10 - min_turns_left) * 0.2
        
        # Egg difference is the most important factor, scaled by endgame factor
        score += egg_diff * 100.0 * endgame_factor
        
        # === MOBILITY ===
        mobility_diff = len(my_moves) - len(opp_moves)
        score += mobility_diff * 5.0
        
        # Reward having more options
        score += len(my_moves) * 2.0
        
        # === TRAPDOOR RISK ===
        my_pos = game_board.chicken_player.get_location()
        risk_here = self.trap_belief.risk(my_pos)
        max_risk_nearby = self.trap_belief.max_risk_in_radius(my_pos, radius=1)
        
        # Penalize high risk positions (trapdoor = -4 eggs for us, +4 for opponent)
        # Effective penalty is 8 egg swing - keep this large for direct risk
        score -= risk_here * 800.0
        
        # Nearby risk penalty reduced from 200 to 120 to avoid being overly
        # terrified of mild risk in the neighborhood
        score -= max_risk_nearby * 120.0
        
        # === CORNER CONTROL ===
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        
        # Being on a corner when we can lay an egg is valuable
        if my_pos in corners and game_board.can_lay_egg():
            score += 30.0  # Corner eggs worth 3x
        
        # Being near a corner is somewhat valuable, but don't over-prioritize
        # Reduced from 2.0 to 1.0 to avoid pulling agent away from good positions
        my_corner_dist = min(abs(my_pos[0] - c[0]) + abs(my_pos[1] - c[1]) for c in corners)
        score -= my_corner_dist * 1.0
        
        # === POSITIONAL CONTROL ===
        # Being more central is generally good (more mobility options)
        center = (3.5, 3.5)
        my_center_dist = abs(my_pos[0] - center[0]) + abs(my_pos[1] - center[1])
        score -= my_center_dist * 1.0
        
        # === TURN/TEMPO ===
        turn_diff = game_board.turns_left_player - game_board.turns_left_enemy
        score += turn_diff * 5.0
        
        # === TURD STRATEGY ===
        my_turds_left = game_board.chicken_player.get_turds_left()
        opp_turds_left = game_board.chicken_enemy.get_turds_left()
        
        # Having turds available is valuable for blocking
        score += my_turds_left * 3.0
        score -= opp_turds_left * 3.0
        
        # === DISTANCE TO OPPONENT ===
        opp_pos = game_board.chicken_enemy.get_location()
        chicken_distance = abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1])
        
        # Being close to opponent can be good for blocking/interference
        # but not too close (need distance 2+ for turds)
        if chicken_distance == 2:
            score += 5.0  # Good turd-dropping distance
        elif chicken_distance < 2:
            score -= 5.0  # Too close
        
        return score
    
    def evaluate(self, game_board: board.Board) -> float:
        """
        Evaluate board from current player's perspective.
        
        If a value net is present, combines it with heuristic.
        Otherwise, just returns heuristic value.
        
        Args:
            game_board: Board state to evaluate
            
        Returns:
            Evaluation score (higher is better for current player)
        """
        heuristic_value = self.heuristic(game_board)
        
        if not self.use_value_net or self.value_net is None:
            return heuristic_value
        
        # Use value network if available
        try:
            features = extract_features(game_board, self.trap_belief)
            features_tensor = torch.from_numpy(features).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                net_value = self.value_net(features_tensor).item()
            
            # Scale network output ([-1, 1]) to reasonable range
            net_value *= 100.0
            
            # Combine heuristic and network (weighted average)
            # Give more weight to heuristic for terminal states
            if abs(heuristic_value) > 1000:
                # Terminal or near-terminal: trust heuristic more
                combined_value = 0.9 * heuristic_value + 0.1 * net_value
            else:
                # Normal position: blend equally
                combined_value = 0.5 * heuristic_value + 0.5 * net_value
            
            return combined_value
        except Exception:
            # If network evaluation fails, fall back to heuristic
            return heuristic_value

