"""
search_mcts.py - Monte Carlo Tree Search with UCT for MaxBeater

Implements MCTS with:
- UCT (Upper Confidence bounds for Trees) for node selection
- Value network evaluation at leaf nodes (no random rollouts)
- Perspective flipping for adversarial search
- Time-bounded search with iterative simulations
"""

from __future__ import annotations
from typing import Callable, Any, Dict, List, Tuple, Optional
import time
import math
import numpy as np
from game import board, enums
from .evaluator import Evaluator
from .belief import TrapdoorBelief


class MCTSNode:
    """
    Node in the MCTS tree.
    
    Represents a game state and tracks statistics for UCT:
    - N: visit count
    - W: total value accumulated
    - Q: average value (W/N)
    - prior: initial probability for this move (for PUCT)
    """
    
    def __init__(
        self,
        game_board: board.Board,
        parent: Optional[MCTSNode] = None,
        prior: float = 1.0,
        move_from_parent: Optional[Tuple] = None
    ):
        """
        Initialize MCTS node.
        
        Args:
            game_board: Board state at this node
            parent: Parent node (None for root)
            prior: Prior probability (for PUCT formula)
            move_from_parent: Move that led to this node from parent
        """
        self.board = game_board
        self.parent = parent
        self.prior = prior
        self.move_from_parent = move_from_parent
        
        # MCTS statistics
        self.children: Dict[Tuple, MCTSNode] = {}
        self.N = 0  # Visit count
        self.W = 0.0  # Total value
        self.Q = 0.0  # Average value (W/N)
        
        # Expansion state
        self.is_expanded = False
        self.is_terminal = game_board.is_game_over() if game_board else False
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not yet expanded)."""
        return not self.is_expanded or self.is_terminal
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate UCB score for this node.
        
        Formula: Q + c_puct * P * sqrt(N_parent) / (1 + N)
        
        Args:
            c_puct: Exploration constant
            parent_visits: Parent's visit count
            
        Returns:
            UCB score
        """
        if self.N == 0:
            # Unvisited node: high exploration bonus
            return self.Q + c_puct * self.prior * math.sqrt(parent_visits + 1)
        else:
            exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.N)
            return self.Q + exploration


class MCTSSearch:
    """
    MCTS search engine for selecting best moves.
    
    Uses UCT for selection, value network for evaluation, and time-bounded search.
    """
    
    def __init__(
        self,
        evaluator: Evaluator,
        trap_belief: TrapdoorBelief,
        rng: np.random.Generator
    ):
        """
        Initialize MCTS search engine.
        
        Args:
            evaluator: Board evaluator (heuristic + value model)
            trap_belief: Trapdoor belief tracker
            rng: Random number generator
        """
        self.evaluator = evaluator
        self.trap_belief = trap_belief
        self.rng = rng
        
        # MCTS hyperparameters
        self.c_puct = 1.5  # Exploration constant (higher = more exploration)
        self.max_simulations = 2000  # Maximum simulations per move
        self.time_safety_margin = 3.0  # Reserve time (seconds)
        self.min_time_per_move = 0.2  # Minimum thinking time
    
    def _filter_moves_early_game(
        self,
        game_board: board.Board,
        moves: List[Tuple[enums.Direction, enums.MoveType]]
    ) -> List[Tuple[enums.Direction, enums.MoveType]]:
        """
        Filter out clearly bad early-game moves, especially TURD spam.
        
        Strategy:
        - If no eggs have been laid by either player and we still have many turns left,
          strongly discourage TURD moves.
        - Only filter TURD moves if there is at least one non-TURD move available.
        """
        if not moves:
            return moves
        
        my_eggs = game_board.chicken_player.get_eggs_laid()
        enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
        total_eggs = my_eggs + enemy_eggs
        
        # Use turns_left_player from board if available, defaulting to 40 otherwise.
        turns_left = getattr(game_board, "turns_left_player", 40)
        
        # Early game: no eggs yet and plenty of turns left.
        if total_eggs == 0 and turns_left > 20:
            non_turd_moves = [
                m for m in moves
                if m[1] != enums.MoveType.TURD
            ]
            # Only filter if we have at least one non-TURD move.
            if non_turd_moves:
                return non_turd_moves
        
        return moves
    
    def choose_move(
        self,
        game_board: board.Board,
        time_left: Callable[[], float],
        turns_left: Optional[int] = None
    ) -> Optional[Tuple[enums.Direction, enums.MoveType]]:
        """
        Choose best move using MCTS.
        
        Args:
            game_board: Current board state
            time_left: Function returning remaining time in seconds
            turns_left: Estimated turns remaining
            
        Returns:
            Best move as (Direction, MoveType) tuple, or None if no valid moves
        """
        valid_moves = game_board.get_valid_moves(enemy=False)
        
        # Filter out bad early-game moves (e.g., TURD spam)
        valid_moves = self._filter_moves_early_game(game_board, valid_moves)
        
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            # Only one move: return immediately
            return valid_moves[0]
        
        # Calculate time budget for this move
        remaining_time = time_left()
        time_budget = self._calculate_time_budget(remaining_time, turns_left)
        
        # Create root node
        root = MCTSNode(game_board, parent=None)
        
        # Expand root immediately
        self._expand_node(root)
        
        # Run MCTS simulations
        start_time = time.time()
        simulations = 0
        
        while simulations < self.max_simulations:
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                break
            
            # Check remaining time
            if time_left() < self.time_safety_margin:
                break
            
            # Run one simulation
            self._run_simulation(root, time_left)
            simulations += 1
        
        # Select best move based on visit counts
        if not root.children:
            # No children expanded (shouldn't happen)
            return valid_moves[0]
        
        best_move = self._select_best_move(root)
        
        # Debug info
        elapsed = time.time() - start_time
        print(f"[MCTS] {simulations} simulations in {elapsed:.2f}s, selected {best_move}")
        
        return best_move
    
    def _calculate_time_budget(
        self,
        remaining_time: float,
        turns_left: Optional[int]
    ) -> float:
        """
        Calculate time budget for current move.
        
        Strategy:
        - Allocate time proportional to remaining turns
        - Reserve safety margin
        - Cap at max/min per move
        """
        if turns_left is None or turns_left <= 0:
            turns_left = 20  # Conservative estimate
        
        # Allocate time: (remaining - safety) / turns_left
        usable_time = max(remaining_time - self.time_safety_margin, 1.0)
        time_per_move = usable_time / turns_left
        
        # Clamp to reasonable range
        time_budget = max(self.min_time_per_move, min(time_per_move, 10.0))
        
        return time_budget
    
    def _run_simulation(self, root: MCTSNode, time_left: Callable[[], float]) -> None:
        """
        Run one MCTS simulation: Selection -> Expansion -> Evaluation -> Backpropagation.
        
        Args:
            root: Root node of search tree
            time_left: Function to check remaining time
        """
        # === SELECTION: traverse tree using UCT ===
        node = root
        path: List[MCTSNode] = [node]
        
        while not node.is_leaf() and node.children:
            # Check time
            if time_left() < self.time_safety_margin:
                return
            
            # Select best child using UCT
            node = self._select_child_uct(node)
            path.append(node)
        
        # === EXPANSION: expand leaf node ===
        if not node.is_terminal and not node.is_expanded:
            self._expand_node(node)
            
            # If expansion created children, select one
            if node.children:
                node = self._select_child_uct(node)
                path.append(node)
        
        # === EVALUATION: evaluate leaf position ===
        value = self.evaluator.evaluate(node.board)
        
        # Normalize value to roughly [-1, 1] for stable MCTS
        # Heuristic values are in range ~[-10000, 10000]
        normalized_value = np.tanh(value / 2000.0)
        
        # === BACKPROPAGATION: update statistics ===
        # Note: value alternates sign at each level (minimax perspective)
        current_value = normalized_value
        for n in reversed(path):
            n.N += 1
            n.W += current_value
            n.Q = n.W / n.N
            current_value = -current_value  # Flip sign for opponent
    
    def _expand_node(self, node: MCTSNode) -> None:
        """
        Expand a node by creating children for all valid moves.
        
        IMPORTANT: After applying a move, we must reverse perspective
        so that the child node represents the opponent's turn.
        
        Args:
            node: Node to expand
        """
        if node.is_terminal or node.is_expanded:
            return
        
        valid_moves = node.board.get_valid_moves(enemy=False)
        
        # Filter out bad early-game moves (e.g., TURD spam)
        valid_moves = self._filter_moves_early_game(node.board, valid_moves)
        
        if not valid_moves:
            node.is_expanded = True
            return
        
        # Compute context for move-type biasing
        my_eggs = node.board.chicken_player.get_eggs_laid()
        enemy_eggs = node.board.chicken_enemy.get_eggs_laid()
        total_eggs = my_eggs + enemy_eggs
        turns_left = getattr(node.board, "turns_left_player", 40)
        
        base_prior = 1.0 / len(valid_moves)
        
        # Create child for each valid move
        for move in valid_moves:
            child_board = self._apply_move(node.board, move)
            
            if child_board is not None:
                direction, move_type = move
                
                # Adjust prior based on move type
                factor = 1.0
                
                # Eggs get a bit more prior mass: we want to explore them more
                if move_type == enums.MoveType.EGG:
                    factor = 1.5
                
                # Early turds get less prior mass
                if move_type == enums.MoveType.TURD and total_eggs == 0 and turns_left > 20:
                    factor = 0.5
                
                prior = base_prior * factor
                
                child = MCTSNode(
                    game_board=child_board,
                    parent=node,
                    prior=prior,
                    move_from_parent=move
                )
                node.children[move] = child
        
        node.is_expanded = True
    
    def _select_child_uct(self, node: MCTSNode) -> MCTSNode:
        """
        Select child with highest UCT score.
        
        Args:
            node: Parent node
            
        Returns:
            Selected child node
        """
        if not node.children:
            return node
        
        best_score = -math.inf
        best_child = None
        
        parent_visits = node.N
        
        for child in node.children.values():
            score = child.ucb_score(self.c_puct, parent_visits)
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child if best_child else node
    
    def _select_best_move(self, root: MCTSNode) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Select best move from root based on visit counts.
        
        Strategy: Choose move with highest visit count (most explored).
        
        Args:
            root: Root node
            
        Returns:
            Best move
        """
        if not root.children:
            # Fallback: random valid move
            valid_moves = root.board.get_valid_moves(enemy=False)
            return valid_moves[0] if valid_moves else (enums.Direction.UP, enums.MoveType.PLAIN)
        
        best_move = None
        best_visits = -1
        
        for move, child in root.children.items():
            if child.N > best_visits:
                best_visits = child.N
                best_move = move
        
        return best_move
    
    def _apply_move(
        self,
        game_board: board.Board,
        move: Tuple[enums.Direction, enums.MoveType]
    ) -> Optional[board.Board]:
        """
        Apply a move and return new board state.
        
        CRITICAL: This function handles perspective correctly:
        1. Clone the board
        2. Apply the move (from current player's perspective)
        3. Reverse perspective so child represents opponent's turn
        
        Args:
            game_board: Current board
            move: (Direction, MoveType) to apply
            
        Returns:
            New board state with perspective reversed, or None if invalid
        """
        try:
            # Use forecast_move if available, otherwise manual clone + apply
            direction, move_type = move
            
            # forecast_move clones the board and applies the move
            new_board = game_board.forecast_move(direction, move_type, check_ok=True)
            
            if new_board is None:
                return None
            
            # IMPORTANT: Reverse perspective so that in the child node,
            # chicken_player represents the opponent (who moves next)
            new_board.reverse_perspective()
            
            return new_board
            
        except Exception as e:
            print(f"[MCTS] Error applying move {move}: {e}")
            return None

