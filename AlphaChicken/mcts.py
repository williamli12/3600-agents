"""
mcts.py - Monte Carlo Tree Search with UCT for AlphaChicken

Implements MCTS with neural network evaluation and UCT selection.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from game import board, enums

try:
    from .model import AlphaChickenNet, board_to_tensor, index_to_move, move_to_index, get_valid_move_mask
except ImportError:
    from model import AlphaChickenNet, board_to_tensor, index_to_move, move_to_index, get_valid_move_mask


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, game_board: board.Board, parent: Optional['MCTSNode'] = None, 
                 prior_prob: float = 0.0, move: Optional[Tuple] = None):
        """
        Initialize MCTS node.
        
        Args:
            game_board: Board state at this node
            parent: Parent node (None for root)
            prior_prob: Prior probability from policy network
            move: Move that led to this node from parent
        """
        self.board = game_board
        self.parent = parent
        self.move = move  # Move that led to this state
        self.prior = prior_prob
        
        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Tuple, MCTSNode] = {}
        
        # Derived properties
        self.is_expanded = False
        
    def value(self) -> float:
        """Average value (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def uct_score(self, c_puct: float = 1.5) -> float:
        """
        UCT score for this node.
        
        U = c_puct * P * sqrt(N_parent) / (1 + N)
        Score = Q + U
        """
        if self.parent is None:
            return 0.0
        
        u = c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value() + u
    
    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """Select child with highest UCT score."""
        return max(self.children.values(), key=lambda child: child.uct_score(c_puct))
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal game state."""
        return self.board.is_game_over()


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""
    
    def __init__(self, model: AlphaChickenNet, trap_belief=None, c_puct: float = 1.5, 
                 device: str = 'cpu'):
        """
        Initialize MCTS.
        
        Args:
            model: Neural network for evaluation
            trap_belief: TrapdoorBelief object for state representation
            c_puct: Exploration constant
            device: PyTorch device
        """
        self.model = model
        self.model.eval()
        self.trap_belief = trap_belief
        self.c_puct = c_puct
        self.device = device
        
    def search(self, root_board: board.Board, num_simulations: int = 100) -> MCTSNode:
        """
        Run MCTS from root board state.
        
        Args:
            root_board: Starting board state
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            Root node with updated statistics
        """
        root = MCTSNode(root_board.get_copy())
        
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree using UCT
            while node.is_expanded and not node.is_terminal():
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Expansion and Evaluation
            if not node.is_terminal():
                value = self._expand_and_evaluate(node)
            else:
                # Terminal node: get true game value
                value = self._get_terminal_value(node.board)
            
            # Backpropagation: update all nodes in search path
            self._backpropagate(search_path, value)
        
        return root
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand node and evaluate with neural network.
        
        Args:
            node: Node to expand
            
        Returns:
            Value estimate from current player's perspective
        """
        # Get valid moves
        valid_moves = node.board.get_valid_moves()
        
        if len(valid_moves) == 0:
            # No valid moves - game over
            node.is_expanded = True
            return self._get_terminal_value(node.board)
        
        # Evaluate position with neural network
        with torch.no_grad():
            state_tensor = board_to_tensor(node.board, self.trap_belief).to(self.device)
            policy_logits, value = self.model(state_tensor)
            
            # Convert to numpy
            policy_probs = torch.exp(policy_logits).cpu().numpy()[0]  # Shape: (12,)
            value_estimate = value.cpu().numpy()[0, 0]  # Scalar
        
        # Mask invalid moves and renormalize
        valid_mask = get_valid_move_mask(node.board)
        masked_probs = policy_probs * valid_mask
        
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # Fallback: uniform over valid moves
            masked_probs = valid_mask / valid_mask.sum()
        
        # Create child nodes
        for direction, move_type in valid_moves:
            move_index = move_to_index(direction, move_type)
            prior_prob = masked_probs[move_index]
            
            # Simulate move
            child_board = node.board.forecast_move(direction, move_type, check_ok=False)
            
            if child_board is not None:
                # Flip perspective for opponent
                child_board.reverse_perspective()
                
                # Create child node
                child = MCTSNode(child_board, parent=node, prior_prob=prior_prob, 
                               move=(direction, move_type))
                node.children[(direction, move_type)] = child
        
        node.is_expanded = True
        return value_estimate
    
    def _get_terminal_value(self, game_board: board.Board) -> float:
        """
        Get true value for terminal state.
        
        Returns +1 for win, -1 for loss, 0 for draw (from current player's perspective).
        """
        if not game_board.is_game_over():
            return 0.0
        
        winner = game_board.get_winner()
        
        if winner == enums.Result.PLAYER:
            return 1.0
        elif winner == enums.Result.ENEMY:
            return -1.0
        else:  # TIE
            return 0.0
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """
        Backpropagate value up the tree.
        
        Value flips sign at each level (minimax style).
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for parent (opponent's perspective)
    
    def get_action_probs(self, root: MCTSNode, temperature: float = 1.0) -> np.ndarray:
        """
        Get move probabilities from MCTS visit counts.
        
        Args:
            root: Root node after search
            temperature: Temperature parameter (0 = deterministic, 1 = stochastic)
            
        Returns:
            Array of shape (12,) with move probabilities
        """
        probs = np.zeros(12, dtype=np.float32)
        
        for move, child in root.children.items():
            index = move_to_index(move[0], move[1])
            probs[index] = child.visit_count
        
        if temperature == 0:
            # Deterministic: choose most visited
            best_index = np.argmax(probs)
            probs = np.zeros(12, dtype=np.float32)
            probs[best_index] = 1.0
        else:
            # Apply temperature
            if probs.sum() > 0:
                probs = probs ** (1.0 / temperature)
                probs = probs / probs.sum()
            else:
                # No children (shouldn't happen)
                probs = np.ones(12) / 12
        
        return probs
    
    def get_best_move(self, root: MCTSNode) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Get move with highest visit count.
        
        Args:
            root: Root node after search
            
        Returns:
            (Direction, MoveType) tuple
        """
        if len(root.children) == 0:
            # No children - return random valid move
            valid_moves = root.board.get_valid_moves()
            if valid_moves:
                return valid_moves[0]
            return (enums.Direction.UP, enums.MoveType.PLAIN)
        
        best_child = max(root.children.items(), key=lambda item: item[1].visit_count)
        return best_child[0]  # Return the move

