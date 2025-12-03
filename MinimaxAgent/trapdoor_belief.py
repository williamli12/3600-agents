# trapdoor_belief.py
"""
Enhanced Bayesian belief tracking for trapdoor locations.
Uses sensor data to maintain probability distributions over possible trapdoor positions.
"""

from typing import Dict, Tuple, List
import numpy as np
from game import board
from game.game_map import prob_hear, prob_feel

Coord = Tuple[int, int]


class TrapdoorBelief:
    """
    Manages Bayesian belief state for two trapdoor locations.
    Trapdoors are constrained by parity: one on even squares, one on odd squares.
    """
    
    def __init__(self, board_size: int = 8):
        """
        Initialize belief distributions for two trapdoors.
        
        Args:
            board_size: Size of the game board (default 8x8)
        """
        self.board_size = board_size
        
        # Separate beliefs for even and odd parity trapdoors
        self.belief_even: Dict[Coord, float] = {}  # Even parity: (x+y) % 2 == 0
        self.belief_odd: Dict[Coord, float] = {}   # Odd parity: (x+y) % 2 == 1
        
        # Track observation history for advanced inference
        self.observations: List[Tuple[Coord, List[Tuple[bool, bool]]]] = []
        
        self._init_priors()
        
    def _init_priors(self) -> None:
        """
        Initialize prior probability distributions based on game rules.
        
        Trapdoors are more likely near the center according to game mechanics:
        - Edge squares (layer 0-1): Very unlikely
        - Layer 2: Medium probability
        - Center (layer 3+): High probability
        """
        for x in range(self.board_size):
            for y in range(self.board_size):
                # Calculate distance from nearest edge
                dist_from_edge = min(x, y, self.board_size - 1 - x, self.board_size - 1 - y)
                
                # Assign prior weight based on distance from edge
                if dist_from_edge <= 1:
                    weight = 0.001  # Very low prior for edges
                elif dist_from_edge == 2:
                    weight = 1.0    # Medium prior
                else:  # dist_from_edge >= 3
                    weight = 2.5    # High prior for center
                
                # Assign to appropriate parity belief
                parity = (x + y) % 2
                if parity == 0:
                    self.belief_even[(x, y)] = weight
                else:
                    self.belief_odd[(x, y)] = weight
        
        # Normalize to create valid probability distributions
        self._normalize(self.belief_even)
        self._normalize(self.belief_odd)
    
    def _normalize(self, belief: Dict[Coord, float]) -> None:
        """Normalize a belief distribution to sum to 1."""
        total = sum(belief.values())
        if total > 0:
            for coord in belief:
                belief[coord] /= total
    
    def update(self, game_board: board.Board, sensor_data: List[Tuple[bool, bool]]) -> None:
        """
        Update both trapdoor belief distributions using Bayes' Rule.
        
        Args:
            game_board: Current game board state
            sensor_data: [(heard_even, felt_even), (heard_odd, felt_odd)]
                        Sensor readings for even and odd parity trapdoors
        """
        position = game_board.chicken_player.get_location()
        
        # Store observation for potential replay
        self.observations.append((position, sensor_data))
        
        # Update even parity trapdoor (index 0)
        heard_even, felt_even = sensor_data[0]
        self._bayesian_update(self.belief_even, position, heard_even, felt_even)
        
        # Update odd parity trapdoor (index 1)
        heard_odd, felt_odd = sensor_data[1]
        self._bayesian_update(self.belief_odd, position, heard_odd, felt_odd)
        
        # Account for found trapdoors (if any marked on board)
        self._update_from_found_trapdoors(game_board)
    
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
            position: Current player position where sensors were read
            heard: Whether player heard this trapdoor
            felt: Whether player felt this trapdoor
        """
        epsilon = 1e-9  # Small value to avoid exact zeros
        
        for coord in prob_grid:
            # Compute likelihood of observation given trapdoor at this coord
            p_hear = self._prob_hear(position, coord)
            p_feel = self._prob_feel(position, coord)
            
            # Clamp probabilities to avoid numerical issues
            p_hear = np.clip(p_hear, epsilon, 1.0 - epsilon)
            p_feel = np.clip(p_feel, epsilon, 1.0 - epsilon)
            
            # Compute likelihood of hearing observation
            if heard:
                likelihood_hear = p_hear
            else:
                likelihood_hear = 1.0 - p_hear
            
            # Compute likelihood of feeling observation
            if felt:
                likelihood_feel = p_feel
            else:
                likelihood_feel = 1.0 - p_feel
            
            # Combined likelihood (assume independence of hear/feel)
            likelihood = likelihood_hear * likelihood_feel
            
            # Bayesian update: posterior ∝ prior * likelihood
            prob_grid[coord] *= likelihood
        
        # Normalize to maintain valid probability distribution
        self._normalize(prob_grid)
    
    def _prob_hear(self, position: Coord, trap_coord: Coord) -> float:
        """
        Probability of hearing a trapdoor at trap_coord from position.
        Uses game-defined probability function.
        """
        dx = abs(position[0] - trap_coord[0])
        dy = abs(position[1] - trap_coord[1])
        return prob_hear(dx, dy)
    
    def _prob_feel(self, position: Coord, trap_coord: Coord) -> float:
        """
        Probability of feeling a trapdoor at trap_coord from position.
        Uses game-defined probability function.
        """
        dx = abs(position[0] - trap_coord[0])
        dy = abs(position[1] - trap_coord[1])
        return prob_feel(dx, dy)
    
    def _update_from_found_trapdoors(self, game_board: board.Board) -> None:
        """
        Update beliefs based on trapdoors that have been found/triggered.
        Set probability to 1.0 for found trapdoors, 0.0 for others.
        """
        found_trapdoors = game_board.found_trapdoors
        
        if not found_trapdoors:
            return
        
        for trap_loc in found_trapdoors:
            parity = (trap_loc[0] + trap_loc[1]) % 2
            
            if parity == 0:
                # Even parity trapdoor found
                for coord in self.belief_even:
                    self.belief_even[coord] = 1.0 if coord == trap_loc else 0.0
            else:
                # Odd parity trapdoor found
                for coord in self.belief_odd:
                    self.belief_odd[coord] = 1.0 if coord == trap_loc else 0.0
    
    def risk(self, coord: Coord) -> float:
        """
        Return combined probability that any trapdoor is at this coord.
        
        Args:
            coord: Position to check
            
        Returns:
            Combined probability (sum of both trapdoor probabilities)
        """
        risk_even = self.belief_even.get(coord, 0.0)
        risk_odd = self.belief_odd.get(coord, 0.0)
        return risk_even + risk_odd
    
    def max_risk_in_radius(self, center: Coord, radius: int = 1) -> float:
        """
        Return maximum trapdoor risk within Manhattan distance of center.
        
        Args:
            center: Center position
            radius: Manhattan distance radius
            
        Returns:
            Maximum risk found in the area
        """
        max_risk = 0.0
        cx, cy = center
        
        for x in range(max(0, cx - radius), min(self.board_size, cx + radius + 1)):
            for y in range(max(0, cy - radius), min(self.board_size, cy + radius + 1)):
                manhattan_dist = abs(x - cx) + abs(y - cy)
                if manhattan_dist <= radius:
                    max_risk = max(max_risk, self.risk((x, y)))
        
        return max_risk
    
    def expected_risk(self) -> float:
        """
        Return expected value of trapdoor risk across all positions.
        Useful for strategic decision-making.
        """
        total_risk = 0.0
        count = 0
        
        for coord in self.belief_even:
            total_risk += self.belief_even[coord]
            count += 1
        
        for coord in self.belief_odd:
            total_risk += self.belief_odd[coord]
            count += 1
        
        return total_risk / count if count > 0 else 0.0
    
    def get_most_likely_positions(self, n: int = 3) -> List[Tuple[Coord, float]]:
        """
        Get the n most likely positions for trapdoors.
        
        Args:
            n: Number of top positions to return
            
        Returns:
            List of (coord, probability) tuples
        """
        all_risks = [(coord, self.risk(coord)) for coord in 
                     list(self.belief_even.keys()) + list(self.belief_odd.keys())]
        all_risks.sort(key=lambda x: x[1], reverse=True)
        return all_risks[:n]

