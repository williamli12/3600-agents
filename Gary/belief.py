# belief.py
"""
Bayesian belief tracking for trapdoor locations using sensor data.
Maintains probability distributions over possible trapdoor positions.
"""

from typing import Dict, Tuple, List
from game import board
from game.game_map import prob_hear, prob_feel

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
        """
        for x in range(self.board_size):
            for y in range(self.board_size):
                # Distance from nearest edge
                dist_from_edge = min(x, y, self.board_size - 1 - x, self.board_size - 1 - y)
                
                # Assign weight based on layer
                if dist_from_edge <= 1:
                    weight = 0.001  # Very low prior for edges
                elif dist_from_edge == 2:
                    weight = 1.0  # Layer 2
                elif dist_from_edge >= 3:
                    weight = 2.0  # Center (Layer 3+)
                else:
                    weight = 0.001
                
                # Assign to appropriate trapdoor based on parity
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
        
        # Handle explicitly found trapdoors
        for trap_loc in board.found_trapdoors:
            self._set_certainty(trap_loc)

    def _set_certainty(self, loc: Coord):
        """Set probability to 1.0 for a known trapdoor location."""
        if (loc[0] + loc[1]) % 2 == 0:
            # Clear all others in this distribution
            for k in self.belief_trap1:
                self.belief_trap1[k] = 0.0
            self.belief_trap1[loc] = 1.0
        else:
            for k in self.belief_trap2:
                self.belief_trap2[k] = 0.0
            self.belief_trap2[loc] = 1.0

    def _bayesian_update(
        self, 
        prob_grid: Dict[Coord, float], 
        position: Coord, 
        heard: bool, 
        felt: bool
    ) -> None:
        """
        Apply Bayesian update to a probability grid.
        """
        epsilon = 1e-6  # Small value to avoid exact zeros
        
        for coord in prob_grid:
            dx = abs(position[0] - coord[0])
            dy = abs(position[1] - coord[1])
            
            p_hear = prob_hear(dx, dy)
            p_feel = prob_feel(dx, dy)
            
            # Clamp probabilities
            p_hear = max(epsilon, min(1 - epsilon, p_hear))
            p_feel = max(epsilon, min(1 - epsilon, p_feel))
            
            # Compute likelihood
            likelihood_hear = p_hear if heard else (1.0 - p_hear)
            likelihood_feel = p_feel if felt else (1.0 - p_feel)
            
            # Combined likelihood (assume independence)
            likelihood = likelihood_hear * likelihood_feel
            
            # Bayesian update
            prob_grid[coord] *= likelihood
        
        # Normalize
        self._normalize(prob_grid)
    
    def risk(self, coord: Coord) -> float:
        """
        Return combined probability that any trapdoor is at this coord.
        """
        risk1 = self.belief_trap1.get(coord, 0.0)
        risk2 = self.belief_trap2.get(coord, 0.0)
        return risk1 + risk2

