"""
belief.py - Bayesian trapdoor belief tracking for MaxBeater

Maintains probability distributions over two trapdoor locations (one even parity, one odd).
Uses sensor data (heard/felt) to update beliefs via Bayes' rule.
"""

from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
from game import board

Coord = Tuple[int, int]


class TrapdoorBelief:
    """
    Bayesian belief tracker for two trapdoors on an 8x8 board.
    
    Game rules:
    - 2 trapdoors total: one on even-parity squares (x+y even), one on odd-parity
    - More likely in center rings (2-3 from edge), less likely on edges/corners
    - Sensor probabilities based on Manhattan distance:
        hear: 0.5 (adj), 0.25 (diag), 0.1 (dist=2), 0 (>2 or dist=2 diag)
        feel: 0.3 (adj), 0.15 (diag), 0 (>1)
    """
    
    def __init__(self, board_size: int = 8):
        self.board_size = board_size
        # Separate beliefs for even and odd parity trapdoors
        self.belief_even: Dict[Coord, float] = {}
        self.belief_odd: Dict[Coord, float] = {}
        self._init_priors()
    
    def _init_priors(self) -> None:
        """
        Initialize prior beliefs based on game trapdoor placement rules.
        
        Trapdoors are placed with weights:
        - Ring 0-1 (edges): weight 0 (impossible)
        - Ring 2: weight 1.0
        - Ring 3 (center 2x2): weight 2.0
        
        We initialize with small epsilon on edges (not true 0) to avoid
        division issues, and normalize.
        """
        for y in range(self.board_size):
            for x in range(self.board_size):
                # Distance from nearest edge
                dist_from_edge = min(x, y, self.board_size - 1 - x, self.board_size - 1 - y)
                
                # Weight based on ring
                if dist_from_edge < 2:
                    weight = 0.001  # Small epsilon, trapdoors rarely spawn here
                elif dist_from_edge == 2:
                    weight = 1.0
                else:  # dist_from_edge >= 3 (center region)
                    weight = 2.0
                
                # Assign to appropriate parity belief
                coord = (x, y)
                if (x + y) % 2 == 0:
                    self.belief_even[coord] = weight
                else:
                    self.belief_odd[coord] = weight
        
        # Normalize to proper probability distributions
        self._normalize()
    
    def _normalize(self) -> None:
        """Normalize both belief distributions to sum to 1."""
        total_even = sum(self.belief_even.values())
        total_odd = sum(self.belief_odd.values())
        
        if total_even > 0:
            for coord in self.belief_even:
                self.belief_even[coord] /= total_even
        
        if total_odd > 0:
            for coord in self.belief_odd:
                self.belief_odd[coord] /= total_odd
    
    def update(self, game_board: board.Board, sensor_data: List[Tuple[bool, bool]]) -> None:
        """
        Update beliefs using Bayes' rule given sensor observations.
        
        Args:
            game_board: Current board state
            sensor_data: [(heard_trap0, felt_trap0), (heard_trap1, felt_trap1)]
                        trap0 is even-parity, trap1 is odd-parity
        """
        player_pos = game_board.chicken_player.get_location()
        found_traps = game_board.found_trapdoors
        
        # Update even-parity trapdoor belief (trap 0)
        if len(sensor_data) > 0:
            heard, felt = sensor_data[0]
            self._bayesian_update(self.belief_even, player_pos, heard, felt, found_traps)
        
        # Update odd-parity trapdoor belief (trap 1)
        if len(sensor_data) > 1:
            heard, felt = sensor_data[1]
            self._bayesian_update(self.belief_odd, player_pos, heard, felt, found_traps)
        
        self._normalize()
    
    def _bayesian_update(
        self,
        belief: Dict[Coord, float],
        player_pos: Coord,
        heard: bool,
        felt: bool,
        found_traps: set
    ) -> None:
        """
        Apply Bayesian update: P(trap at s | observation) ∝ P(obs | trap at s) * P(trap at s)
        """
        for coord in belief.keys():
            # If we've found this trapdoor, set probability to 0
            if coord in found_traps:
                belief[coord] = 0.0
                continue
            
            # Calculate Manhattan distance
            dx = abs(coord[0] - player_pos[0])
            dy = abs(coord[1] - player_pos[1])
            
            # Calculate likelihood P(heard, felt | trapdoor at coord)
            likelihood = self._calc_likelihood(dx, dy, heard, felt)
            
            # Bayesian update: posterior ∝ likelihood × prior
            belief[coord] *= likelihood
    
    def _calc_likelihood(self, dx: int, dy: int, heard: bool, felt: bool) -> float:
        """
        Calculate P(heard, felt | trapdoor at distance (dx, dy)).
        
        Based on game's probability functions:
        - prob_hear(dx, dy)
        - prob_feel(dx, dy)
        """
        # Standing on the trapdoor (would have been found)
        if dx == 0 and dy == 0:
            return 1.0
        
        # Probability of hearing based on distance
        if dx > 2 or dy > 2:
            p_hear = 0.0
        elif dx == 2 and dy == 2:
            p_hear = 0.0
        elif dx == 2 or dy == 2:
            p_hear = 0.1
        elif dx == 1 and dy == 1:
            p_hear = 0.25
        elif dx == 1 or dy == 1:
            p_hear = 0.5
        else:
            p_hear = 0.0
        
        # Probability of feeling based on distance
        if dx > 1 or dy > 1:
            p_feel = 0.0
        elif dx == 1 and dy == 1:
            p_feel = 0.15
        elif dx == 1 or dy == 1:
            p_feel = 0.3
        else:
            p_feel = 0.0
        
        # Calculate likelihood of observation
        # Use small epsilon instead of 0 to avoid complete elimination
        epsilon = 0.01
        
        if heard:
            hear_likelihood = max(p_hear, epsilon)
        else:
            hear_likelihood = max(1.0 - p_hear, epsilon)
        
        if felt:
            feel_likelihood = max(p_feel, epsilon)
        else:
            feel_likelihood = max(1.0 - p_feel, epsilon)
        
        return hear_likelihood * feel_likelihood
    
    def risk(self, coord: Coord) -> float:
        """
        Combined probability that any trapdoor is at coord.
        
        Returns:
            Sum of probabilities from both trapdoor beliefs
        """
        return self.belief_even.get(coord, 0.0) + self.belief_odd.get(coord, 0.0)
    
    def max_risk_in_radius(self, coord: Coord, radius: int = 2) -> float:
        """
        Maximum trapdoor risk among squares within Manhattan radius of coord.
        
        Args:
            coord: Center position
            radius: Manhattan distance radius
            
        Returns:
            Maximum risk value in the neighborhood
        """
        max_risk = 0.0
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                # Check if within radius
                manhattan_dist = abs(x - coord[0]) + abs(y - coord[1])
                if manhattan_dist <= radius:
                    risk_val = self.risk((x, y))
                    max_risk = max(max_risk, risk_val)
        
        return max_risk
    
    def expected_risk_in_radius(self, coord: Coord, radius: int = 2) -> float:
        """
        Expected (average) trapdoor risk within Manhattan radius of coord.
        """
        total_risk = 0.0
        count = 0
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                manhattan_dist = abs(x - coord[0]) + abs(y - coord[1])
                if manhattan_dist <= radius:
                    total_risk += self.risk((x, y))
                    count += 1
        
        return total_risk / count if count > 0 else 0.0
    
    def get_belief_grid(self) -> np.ndarray:
        """
        Get combined belief as 8x8 grid for visualization or feature extraction.
        
        Returns:
            8x8 numpy array with combined trapdoor probabilities
        """
        grid = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                coord = (x, y)
                grid[y, x] = self.risk(coord)
        
        return grid


