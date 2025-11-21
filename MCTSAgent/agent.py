from collections.abc import Callable
from typing import List, Tuple, Dict, Optional
import math
import time
import numpy as np

from game import *

"""
Tournament-Level MCTS Agent with Bayesian Trapdoor Tracking
"""


class TrapdoorTracker:
    """
    Manages Bayesian belief state for trapdoor locations using sensor data.
    """
    
    def __init__(self, map_size: int = 8, max_dist_from_center: int = 2):
        """
        Initialize probability grids for two trapdoors (one per team/parity).
        
        Args:
            map_size: Size of the game board
            max_dist_from_center: Maximum Manhattan distance from center for trapdoors
        """
        self.map_size = map_size
        self.max_dist_from_center = max_dist_from_center
        
        # Two probability grids: one for each trapdoor (Team A and Team B)
        self.prob_trap_A = np.zeros((map_size, map_size))
        self.prob_trap_B = np.zeros((map_size, map_size))
        
        # Initialize with uniform prior weighted towards center
        self._initialize_priors()
    
    def _initialize_priors(self):
        """
        Initialize probability distributions using layer-based weights.
        
        Layer weights (distance from edge):
        - Edge (Layer 0): Weight 0
        - Layer 1: Weight 0
        - Layer 2: Weight 1
        - Center (Layer 3): Weight 2
        """
        # For each cell, calculate layer (distance from nearest edge)
        for x in range(self.map_size):
            for y in range(self.map_size):
                # Distance from nearest edge
                dist_from_edge = min(x, y, self.map_size - 1 - x, self.map_size - 1 - y)
                
                # Assign weight based on layer
                if dist_from_edge <= 1:
                    weight = 0.0  # Edge and Layer 1
                elif dist_from_edge == 2:
                    weight = 1.0  # Layer 2
                elif dist_from_edge >= 3:
                    weight = 2.0  # Center (Layer 3+)
                else:
                    weight = 0.0
                
                # Assign to appropriate trapdoor based on parity
                # Even parity (Team A): (x + y) % 2 == 0
                if (x + y) % 2 == 0:
                    self.prob_trap_A[x, y] = weight
                else:
                    self.prob_trap_B[x, y] = weight
        
        # Normalize to make valid probability distributions
        if self.prob_trap_A.sum() > 0:
            self.prob_trap_A /= self.prob_trap_A.sum()
        if self.prob_trap_B.sum() > 0:
            self.prob_trap_B /= self.prob_trap_B.sum()
    
    def update(self, location: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        """
        Update trapdoor probability distributions using Bayes' Rule.
        
        Args:
            location: Current agent location (x, y)
            sensor_data: [(heard_A, felt_A), (heard_B, felt_B)]
        """
        heard_A, felt_A = sensor_data[0]
        heard_B, felt_B = sensor_data[1]
        
        # Update for Trapdoor A
        self._bayesian_update(self.prob_trap_A, location, heard_A, felt_A)
        
        # Update for Trapdoor B
        self._bayesian_update(self.prob_trap_B, location, heard_B, felt_B)
    
    def _bayesian_update(self, prob_grid: np.ndarray, location: Tuple[int, int], 
                         heard: bool, felt: bool):
        """
        Apply Bayesian update to a probability grid.
        
        P(Trap at x,y | Sensor) = P(Sensor | Trap at x,y) * P(Trap at x,y) / P(Sensor)
        
        Args:
            prob_grid: Probability distribution to update
            location: Agent's current location
            heard: Whether agent heard a sound
            felt: Whether agent felt vibrations
        """
        # Calculate likelihood for each cell
        likelihood = np.zeros_like(prob_grid)
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                if prob_grid[x, y] > 0:  # Only update cells with non-zero prior
                    # Calculate distance from current location
                    delta_x = abs(x - location[0])
                    delta_y = abs(y - location[1])
                    
                    # Get sensor probabilities from game rules
                    p_hear = self._prob_hear(delta_x, delta_y)
                    p_feel = self._prob_feel(delta_x, delta_y)
                    
                    # Calculate P(Sensor | Trap at x,y)
                    p_sensor_given_trap = 1.0
                    
                    if heard:
                        p_sensor_given_trap *= p_hear
                    else:
                        p_sensor_given_trap *= (1.0 - p_hear)
                    
                    if felt:
                        p_sensor_given_trap *= p_feel
                    else:
                        p_sensor_given_trap *= (1.0 - p_feel)
                    
                    likelihood[x, y] = p_sensor_given_trap
        
        # Apply Bayes' Rule: Posterior = Likelihood * Prior
        prob_grid[:] = likelihood * prob_grid
        
        # Normalize
        total = prob_grid.sum()
        if total > 0:
            prob_grid[:] /= total
    
    def _prob_hear(self, delta_x: int, delta_y: int) -> float:
        """Probability of hearing based on distance."""
        # If exactly on the trapdoor, definitely hear it
        if delta_x == 0 and delta_y == 0:
            return 1.0
        if delta_x > 2 or delta_y > 2:
            return 0.0
        if delta_x == 2 and delta_y == 2:
            return 0.0
        if delta_x == 2 or delta_y == 2:
            return 0.1
        if delta_x == 1 and delta_y == 1:
            return 0.25
        if delta_x == 1 or delta_y == 1:
            return 0.5
        return 0.0
    
    def _prob_feel(self, delta_x: int, delta_y: int) -> float:
        """Probability of feeling based on distance."""
        # If exactly on the trapdoor, definitely feel it
        if delta_x == 0 and delta_y == 0:
            return 1.0
        if delta_x > 1 or delta_y > 1:
            return 0.0
        if delta_x == 1 and delta_y == 1:
            return 0.15
        if delta_x == 1 or delta_y == 1:
            return 0.3
        return 0.0
    
    def get_risk(self, location: Tuple[int, int]) -> float:
        """
        Get the probability of a trapdoor being at the given location.
        
        Args:
            location: (x, y) coordinates
            
        Returns:
            Probability between 0.0 and 1.0
        """
        x, y = location
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return 0.0
        
        # Return probability from appropriate grid based on parity
        if (x + y) % 2 == 0:
            return float(self.prob_trap_A[x, y])
        else:
            return float(self.prob_trap_B[x, y])


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    """
    
    def __init__(self, board_state: board.Board, parent: Optional['MCTSNode'] = None, 
                 parent_move: Optional[Tuple] = None):
        """
        Initialize MCTS node.
        
        Args:
            board_state: Game board state
            parent: Parent node in the tree
            parent_move: Move that led from parent to this node
        """
        self.board_state = board_state
        self.parent = parent
        self.parent_move = parent_move
        
        self.children: Dict[Tuple, 'MCTSNode'] = {}
        self.visits = 0
        self.value_sum = 0.0
        
        # Cache valid moves
        self._untried_moves = None
    
    def get_untried_moves(self) -> List[Tuple]:
        """Get list of moves that haven't been expanded yet."""
        if self._untried_moves is None:
            all_moves = self.board_state.get_valid_moves()
            self._untried_moves = [m for m in all_moves if m not in self.children]
        return self._untried_moves
    
    def is_fully_expanded(self) -> bool:
        """Check if all valid moves have been tried."""
        return len(self.get_untried_moves()) == 0
    
    def best_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.
        
        UCB1 = exploitation + exploration
             = (value / visits) + c * sqrt(ln(parent_visits) / visits)
        
        Args:
            exploration_weight: Exploration constant (typically sqrt(2))
            
        Returns:
            Best child node
        """
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                # Prioritize unvisited children
                return child
            
            # UCB1 formula
            exploitation = child.value_sum / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def expand(self, move: Tuple) -> 'MCTSNode':
        """
        Expand node by adding a child for the given move.
        
        Args:
            move: (Direction, MoveType) tuple
            
        Returns:
            New child node
        """
        # Forecast the move and reverse perspective
        new_board = self.board_state.forecast_move(move[0], move[1])
        if new_board is not None:
            new_board.reverse_perspective()
        
        child = MCTSNode(new_board, parent=self, parent_move=move)
        self.children[move] = child
        
        # Remove from untried moves
        if self._untried_moves is not None and move in self._untried_moves:
            self._untried_moves.remove(move)
        
        return child
    
    def get_prioritized_untried_moves(self) -> List[Tuple]:
        """
        Get untried moves prioritized by type: EGG moves first, then PLAIN, then TURD.
        This encourages MCTS to explore egg-laying early.
        """
        untried = self.get_untried_moves()
        if not untried:
            return []
        
        # Separate by move type
        egg_moves = [m for m in untried if m[1] == enums.MoveType.EGG]
        plain_moves = [m for m in untried if m[1] == enums.MoveType.PLAIN]
        turd_moves = [m for m in untried if m[1] == enums.MoveType.TURD]
        
        # Prioritize: EGG > PLAIN > TURD
        return egg_moves + plain_moves + turd_moves
    
    def backpropagate(self, value: float):
        """
        Update node statistics up the tree.
        
        Args:
            value: Value to backpropagate (from perspective of node's player)
        """
        self.visits += 1
        self.value_sum += value
        
        if self.parent is not None:
            # Negate value when passing to parent (alternating players)
            self.parent.backpropagate(-value)


class PlayerAgent:
    """
    MCTS Agent with Bayesian Trapdoor Tracking.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        """
        Initialize the agent.
        
        Args:
            board: Initial game board
            time_left: Callable that returns remaining time
        """
        self.tracker = TrapdoorTracker(board.game_map.MAP_SIZE, 
                                       board.game_map.MAX_TRAPDOOR_DIST_FROM_CENTER)
        self.time_safety_threshold = 0.5
        self.iterations_per_time_check = 100
        
        # Track last move for death detection
        self.last_location: Optional[Tuple[int, int]] = None
        self.last_move: Optional[Tuple[enums.Direction, enums.MoveType]] = None
        self.start_pos: Optional[Tuple[int, int]] = None

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Select best move using MCTS.
        
        Args:
            board: Current game state
            sensor_data: Trapdoor sensor data
            time_left: Callable returning remaining time
            
        Returns:
            Best move (Direction, MoveType)
        """
        current_location = board.chicken_player.get_location()
        
        # Step A: Detect Death (Trapdoor Teleportation)
        if self.last_location is not None and self.last_move is not None and self.start_pos is not None:
            expected_location = self.get_expected_location(self.last_location, self.last_move[0])
            
            # Check if we teleported back to start (died on trapdoor)
            if current_location == self.start_pos and current_location != expected_location:
                # WE DIED! expected_location is a TRAP
                # Force certainty at the trap location
                trap_x, trap_y = expected_location
                
                # Determine which trapdoor grid to update based on parity
                if (trap_x + trap_y) % 2 == 0:
                    # Team A trapdoor (even parity)
                    self.tracker.prob_trap_A[:, :] = 0.0
                    self.tracker.prob_trap_A[trap_x, trap_y] = 1.0
                else:
                    # Team B trapdoor (odd parity)
                    self.tracker.prob_trap_B[:, :] = 0.0
                    self.tracker.prob_trap_B[trap_x, trap_y] = 1.0
        
        # Step B: Store start position on first turn
        if self.start_pos is None:
            self.start_pos = current_location
        
        # Step C: Update trapdoor beliefs with sensor data
        self.tracker.update(current_location, sensor_data)
        
        # Calculate time budget for this move
        time_budget = self.calculate_time_budget(board, time_left())
        start_time = time.time()
        
        # Get valid moves
        valid_moves = board.get_valid_moves()
        if len(valid_moves) == 1:
            best_move = valid_moves[0]
        else:
            # Step D: Initialize MCTS root and run iterations
            root = MCTSNode(board)
            
            # Run MCTS iterations
            iterations = 0
            while True:
                # Check time periodically
                if iterations % self.iterations_per_time_check == 0:
                    elapsed = time.time() - start_time
                    if elapsed >= time_budget or time_left() < self.time_safety_threshold:
                        break
                
                # MCTS iteration
                self.mcts_iteration(root)
                iterations += 1
            
            # Select move with highest visit count
            best_move = self.select_best_move(root)
        
        # Step E: Save current state for next turn's death detection
        self.last_location = current_location
        self.last_move = best_move
        
        return best_move
    
    def get_expected_location(self, location: Tuple[int, int], direction: enums.Direction) -> Tuple[int, int]:
        """
        Calculate the expected location after moving in a direction.
        
        Args:
            location: Current (x, y) position
            direction: Direction to move
            
        Returns:
            Expected (x, y) position after move
        """
        x, y = location
        
        if direction == enums.Direction.UP:
            return (x, y - 1)
        elif direction == enums.Direction.DOWN:
            return (x, y + 1)
        elif direction == enums.Direction.LEFT:
            return (x - 1, y)
        elif direction == enums.Direction.RIGHT:
            return (x + 1, y)
        else:
            return location
    
    def mcts_iteration(self, root: MCTSNode):
        """
        Perform one iteration of MCTS: Selection, Expansion, Evaluation, Backpropagation.
        
        Args:
            root: Root node of the search tree
        """
        # 1. Selection: Traverse tree using UCB1
        node = root
        while not node.board_state.is_game_over() and node.is_fully_expanded():
            if len(node.children) == 0:
                break
            node = node.best_child()
        
        # 2. Expansion: Add one child node (prioritize egg-laying moves)
        if not node.board_state.is_game_over():
            untried_moves = node.get_prioritized_untried_moves()
            if untried_moves:
                move = untried_moves[0]  # Pick first prioritized untried move (EGG moves first!)
                node = node.expand(move)
        
        # 3. Evaluation: Evaluate leaf node
        value = self.evaluate_state(node.board_state)
        
        # 4. Backpropagation: Update statistics up the tree
        node.backpropagate(value)
    
    def evaluate_state(self, board: board.Board) -> float:
        """
        Heuristic evaluation of a board state.
        
        Args:
            board: Game state to evaluate (from current player's perspective)
            
        Returns:
            Normalized value between -1 and 1
        """
        # Terminal state check
        if board.is_game_over():
            winner = board.get_winner()
            if winner == enums.Result.PLAYER:
                return 1.0
            elif winner == enums.Result.ENEMY:
                return -1.0
            else:
                return 0.0
        
        score = 0.0
        
        # Primary: Egg difference (×40 weight - MAXIMUM priority for egg-laying)
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        score += (my_eggs - enemy_eggs) * 40.0
        
        # Get locations
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        
        # CRITICAL: Bonus for being able to lay an egg RIGHT NOW
        if board.can_lay_egg():
            score += 15.0  # Huge bonus for being on an egg-laying square!
        
        # Corner control bonus (corners give 3x eggs)
        corners = [
            (0, 0), (0, 7), (7, 0), (7, 7)
        ]
        
        if my_loc in corners and board.can_lay_egg():
            score += 10.0  # Massive bonus for corner egg opportunities
        if enemy_loc in corners:
            enemy_board = board.get_copy()
            enemy_board.reverse_perspective()
            if enemy_board.can_lay_egg():
                score -= 10.0
        
        # Mobility difference (reduced importance - don't just wander)
        my_moves = len(board.get_valid_moves())
        enemy_board = board.get_copy()
        enemy_board.reverse_perspective()
        enemy_moves = len(enemy_board.get_valid_moves())
        score += (my_moves - enemy_moves) * 0.3
        
        # Encourage being on correct parity squares (where eggs can be laid)
        my_parity = (my_loc[0] + my_loc[1]) % 2
        if my_parity == board.chicken_player.even_chicken:
            score += 2.0  # Small bonus for being on correct parity
        
        # CRITICAL: Trapdoor risk penalties (MINIMAL - don't let fear stop egg-laying)
        # Penalize if current player (me in this perspective) is at risk
        # Reduced to 30.0 (a 10% risk = -3 points, vs +40 for an egg)
        my_trapdoor_risk = self.tracker.get_risk(my_loc)
        score -= my_trapdoor_risk * 30.0
        
        # REWARD if enemy is at risk (good for me!)
        enemy_trapdoor_risk = self.tracker.get_risk(enemy_loc)
        score += enemy_trapdoor_risk * 60.0
        
        # Normalize using tanh to squeeze into [-1, 1]
        normalized_score = math.tanh(score / 70.0)
        
        return normalized_score
    
    def select_best_move(self, root: MCTSNode) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Select the move with the highest visit count.
        
        Args:
            root: Root node of the search tree
            
        Returns:
            Best move
        """
        if not root.children:
            # Fallback to random valid move
            valid_moves = root.board_state.get_valid_moves()
            return valid_moves[0] if valid_moves else (enums.Direction.UP, enums.MoveType.PLAIN)
        
        best_move = None
        best_visits = -1
        
        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
        
        return best_move
    
    def calculate_time_budget(self, board: board.Board, time_remaining: float) -> float:
        """
        Calculate time budget for this move.
        
        Args:
            board: Current game state
            time_remaining: Total time remaining
            
        Returns:
            Time budget in seconds
        """
        turns_played = board.turn_count // 2
        estimated_turns_remaining = max(1, 40 - turns_played)
        
        time_budget = time_remaining / estimated_turns_remaining
        time_budget = min(15.0, max(1.0, time_budget))
        time_budget *= 0.9  # Safety margin
        
        return time_budget
