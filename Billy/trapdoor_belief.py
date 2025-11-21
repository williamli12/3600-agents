# trapdoor_belief.py
"""
Bayesian belief tracking for trapdoor locations using sensor data.
Maintains probability distributions over possible trapdoor positions.
"""

from typing import Dict, Tuple, List
import numpy as np
from game import board

Coord = Tuple[int, int]


class TrapdoorBelief:
    """
    Manages Bayesian belief state for two trapdoor locations.
    Each trapdoor can only be on squares of a specific parity.
    """
    
    def __init__(self, board_size: int = 8):
        """
        Initialize belief distributions for two trapdoors.
        
        Args:
            board_size: Size of the game board (default 8x8)
        """
        self.board_size = board_size
        self.belief_trap1: Dict[Coord, float] = {}
        self.belief_trap2: Dict[Coord, float] = {}
        self._init_priors()
        
    def _init_priors(self) -> None:
        """
        Initialize priors with parity + ring-based weighting.
        
        Layer weights (distance from edge):
        - Edge (Layer 0): Weight 0
        - Layer 1: Weight 0
        - Layer 2: Weight 1
        - Center (Layer 3+): Weight 2
        """
        for x in range(self.board_size):
            for y in range(self.board_size):
                # Distance from nearest edge
                dist_from_edge = min(x, y, self.board_size - 1 - x, self.board_size - 1 - y)
                
                # Assign weight based on layer
                if dist_from_edge <= 1:
                    weight = 0.001  # Very low prior for edges (avoid exact zero)
                elif dist_from_edge == 2:
                    weight = 1.0  # Layer 2
                elif dist_from_edge >= 3:
                    weight = 2.0  # Center (Layer 3+)
                else:
                    weight = 0.001
                
                # Assign to appropriate trapdoor based on parity
                # Even parity (Trap 1): (x + y) % 2 == 0
                # Odd parity (Trap 2): (x + y) % 2 == 1
                if (x + y) % 2 == 0:
                    self.belief_trap1[(x, y)] = weight
                else:
                    self.belief_trap2[(x, y)] = weight
        
        # Normalize to make valid probability distributions
        self._normalize(self.belief_trap1)
        self._normalize(self.belief_trap2)
    
    def _normalize(self, belief: Dict[Coord, float]) -> None:
        """Normalize a belief distribution to sum to 1."""
        total = sum(belief.values())
        if total > 0:
            for coord in belief:
                belief[coord] /= total
    
    def update(self, board: board.Board, sensor_data: List[Tuple[bool, bool]]) -> None:
        """
        Update both trapdoor belief distributions using Bayes' Rule.
        
        Args:
            board: Current game board state
            sensor_data: [(heard_A, felt_A), (heard_B, felt_B)]
        """
        position = board.chicken_player.get_location()
        
        # Update for Trapdoor 1 (even parity)
        heard_1, felt_1 = sensor_data[0]
        self._bayesian_update(self.belief_trap1, position, heard_1, felt_1)
        
        # Update for Trapdoor 2 (odd parity)
        heard_2, felt_2 = sensor_data[1]
        self._bayesian_update(self.belief_trap2, position, heard_2, felt_2)
    
    def _bayesian_update(
        self, 
        prob_grid: Dict[Coord, float], 
        position: Coord, 
        heard: bool, 
        felt: bool
    ) -> None:
        """
        Apply Bayesian update to a probability grid.
        
        P(Trap at x,y | Sensor) = P(Sensor | Trap at x,y) * P(Trap at x,y) / P(Sensor)
        
        Args:
            prob_grid: Probability distribution to update
            position: Current player position
            heard: Whether player heard this trapdoor
            felt: Whether player felt this trapdoor
        """
        epsilon = 1e-6  # Small value to avoid exact zeros
        
        for coord in prob_grid:
            # Compute likelihood of observation given trapdoor at this coord
            p_hear = self._prob_hear(position, coord)
            p_feel = self._prob_feel(position, coord)
            
            # Clamp probabilities to avoid exact 0/1
            p_hear = max(epsilon, min(1 - epsilon, p_hear))
            p_feel = max(epsilon, min(1 - epsilon, p_feel))
            
            # Compute likelihood
            if heard:
                likelihood_hear = p_hear
            else:
                likelihood_hear = 1.0 - p_hear
            
            if felt:
                likelihood_feel = p_feel
            else:
                likelihood_feel = 1.0 - p_feel
            
            # Combined likelihood (assume independence)
            likelihood = likelihood_hear * likelihood_feel
            
            # Bayesian update: posterior ∝ prior * likelihood
            prob_grid[coord] *= likelihood
        
        # Normalize
        self._normalize(prob_grid)
    
    def _prob_hear(self, position: Coord, trap_coord: Coord) -> float:
        """
        Probability of hearing a trapdoor at trap_coord from position.
        
        Based on game rules:
        - Edge-adjacent (side neighbor): 0.5
        - Diagonal-adjacent: 0.25
        - Ring 2 (distance 2): 0.1
        - Far: 0.0
        """
        dx = abs(position[0] - trap_coord[0])
        dy = abs(position[1] - trap_coord[1])
        
        # Use game's prob_hear function if available
        from game.game_map import prob_hear
        return prob_hear(dx, dy)
    
    def _prob_feel(self, position: Coord, trap_coord: Coord) -> float:
        """
        Probability of feeling a trapdoor at trap_coord from position.
        
        Based on game rules:
        - Edge-adjacent (side neighbor): 0.3
        - Diagonal-adjacent: 0.15
        - Ring 2+: 0.0
        """
        dx = abs(position[0] - trap_coord[0])
        dy = abs(position[1] - trap_coord[1])
        
        # Use game's prob_feel function if available
        from game.game_map import prob_feel
        return prob_feel(dx, dy)
    
    def risk(self, coord: Coord) -> float:
        """
        Return combined probability that any trapdoor is at this coord.
        
        Args:
            coord: Position to check
            
        Returns:
            Combined probability (sum of both trapdoor probabilities)
        """
        risk1 = self.belief_trap1.get(coord, 0.0)
        risk2 = self.belief_trap2.get(coord, 0.0)
        return risk1 + risk2
    
    def max_risk_in_radius(self, center: Coord, radius: int = 2) -> float:
        """
        Return maximum trapdoor risk within Manhattan distance of center.
        
        Args:
            center: Center position
            radius: Manhattan distance radius
            
        Returns:
            Maximum risk found
        """
        max_risk = 0.0
        for x in range(max(0, center[0] - radius), min(self.board_size, center[0] + radius + 1)):
            for y in range(max(0, center[1] - radius), min(self.board_size, center[1] + radius + 1)):
                if abs(x - center[0]) + abs(y - center[1]) <= radius:
                    max_risk = max(max_risk, self.risk((x, y)))
        return max_risk

