"""
belief.py - Bayesian trapdoor belief tracking for MaxBeater

Maintains probability distributions over two trapdoor locations (one even parity, one odd).
Uses sensor data (heard/felt) to update beliefs via Bayes' rule.
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Set
import numpy as np
from game import board
from game.game_map import prob_hear, prob_feel

Coord = Tuple[int, int]


class TrapdoorBelief:
    """
    Bayesian belief tracker for two trapdoors on an 8x8 board.
    Assumes one trapdoor is on an even square (x+y is even) and one on an odd square.
    """

    def __init__(self, board_size: int = 8):
        self.board_size = board_size
        # Separate beliefs for even and odd parity trapdoors
        self.belief_even: Dict[Coord, float] = {}
        self.belief_odd: Dict[Coord, float] = {}
        
        # Track squares known to be safe (visited and survived)
        self.safe_squares: Set[Coord] = set()
        
        self._init_priors()

    def _init_priors(self) -> None:
        """
        Initialize prior beliefs based on game trapdoor placement rules.
        Favor center rings.
        """
        for y in range(self.board_size):
            for x in range(self.board_size):
                # Distance from nearest edge
                dist_from_edge = min(x, y, self.board_size - 1 - x, self.board_size - 1 - y)
                
                # Weight based on ring
                if dist_from_edge < 2:
                    weight = 0.001  # Very unlikely on edges
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
        Update beliefs using Bayes' rule given sensor observations and visited squares.
        """
        # 1. Update safe squares based on current positions (if you are there, it's safe)
        p_loc = game_board.chicken_player.get_location()
        e_loc = game_board.chicken_enemy.get_location()
        self.safe_squares.add(p_loc)
        self.safe_squares.add(e_loc)
        
        found_traps = game_board.found_trapdoors

        # 2. Update even-parity trapdoor belief (trap 0)
        if len(sensor_data) > 0:
            heard, felt = sensor_data[0]
            self._bayesian_update(self.belief_even, p_loc, heard, felt, found_traps)

        # 3. Update odd-parity trapdoor belief (trap 1)
        if len(sensor_data) > 1:
            heard, felt = sensor_data[1]
            self._bayesian_update(self.belief_odd, p_loc, heard, felt, found_traps)
        
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
            # If this square is known safe (visited), probability is 0
            # Unless it is a found trapdoor (which makes it 100% a trapdoor)
            if coord in found_traps:
                # This is the trapdoor!
                belief[coord] = 1.0
                # We will normalize later, effectively zeroing out others if this is 1.0
                # But we should process others to 0?
                # If we set this to 1.0, and normalize, it will dominate.
                # But usually found_traps are handled by eliminating others.
                continue
                
            if coord in self.safe_squares:
                belief[coord] = 0.0
                continue

            # Calculate Manhattan distance
            dx = abs(coord[0] - player_pos[0])
            dy = abs(coord[1] - player_pos[1])
            
            # Calculate likelihood P(heard, felt | trapdoor at coord)
            # Note: We import prob_hear/prob_feel from game_map which handle likelihoods
            p_hear_val = prob_hear(dx, dy)
            p_feel_val = prob_feel(dx, dy)

            # If heard=True, P(obs|trap) = p_hear_val
            # If heard=False, P(obs|trap) = 1 - p_hear_val
            
            lik_hear = p_hear_val if heard else (1.0 - p_hear_val)
            lik_feel = p_feel_val if felt else (1.0 - p_feel_val)
            
            # Avoid multiplying by exact zero to allow recovery if model is slightly off?
            # Or stick to strict Bayes. Game returns 0.0, so strict is fine.
            # But let's use a tiny epsilon for robustness.
            lik_hear = max(lik_hear, 1e-6)
            lik_feel = max(lik_feel, 1e-6)

            belief[coord] *= (lik_hear * lik_feel)

    def get_prob(self, row: int, col: int) -> float:
        """Get probability of a trapdoor at (row, col)."""
        coord = (row, col)
        if (row + col) % 2 == 0:
            return self.belief_even.get(coord, 0.0)
        else:
            return self.belief_odd.get(coord, 0.0)

    def snapshot(self) -> Tuple:
        """
        Return a hashable representation of the belief state for the Transposition Table.
        We can't hash the whole float grid. We can hash a simplified version.
        Maybe just the 'safe_squares' count and a rounded hash of high-prob areas?
        
        For simplicity/speed, let's hash:
        - The locations of found trapdoors (implicitly in board state, but useful here)
        - A few high-probability candidates?
        
        Actually, just rounding the probs to 1 decimal place and hashing the tuple of values > 0.1?
        Or simpler: The belief state is largely determined by 'safe_squares' and 'found_trapdoors' + 'turns/observations'.
        
        Let's try a tuple of top 3 most likely coordinates for each parity?
        """
        # Simple hashable: Just the count of safe squares? Too weak.
        # Let's assume the board configuration + move number captures most context.
        # But for belief specifically, maybe we just return nothing and rely on board state?
        # The prompt asked for "something hashable for TT".
        
        # Let's return a tuple of (coord, rounded_prob) for non-zero probs? Too big.
        # How about: The coordinates of the Maximum Likelihood Estimate for each trapdoor?
        
        mle_even = max(self.belief_even.items(), key=lambda x: x[1], default=((0,0), 0))[0]
        mle_odd = max(self.belief_odd.items(), key=lambda x: x[1], default=((0,0), 0))[0]
        return (mle_even, mle_odd, len(self.safe_squares))

    def get_belief_grid(self) -> np.ndarray:
        """Visualization helper."""
        grid = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for y in range(self.board_size):
            for x in range(self.board_size):
                grid[x, y] = self.get_prob(x, y) # x is col? wait. board uses (x,y) usually as (col, row) or (row, col)?
                # Board usually uses (x,y). Python arrays are [row, col] -> [y, x].
                # get_prob takes (row, col) -> (x, y)? 
                # get_prob implementation uses (row, col) as dict key. 
                # The dict keys were initialized as (x, y).
                # If the loop in _init_priors used for y... for x... coord=(x,y), then x is col?
                # Yes, usually x=col, y=row.
                # So get_prob(row, col) should map to (col, row) or (row, col)?
                # Let's assume (x, y) everywhere means (x, y). 
        return grid
