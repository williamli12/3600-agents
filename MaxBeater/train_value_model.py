"""
train_value_model.py - PyTorch training script for value function

This script is ONLY for offline training on PACE cluster.
It will NOT be imported or used during competition runtime.

Training approach:
1. Self-play using MCTS or greedy search to generate training data
2. Collect (state, outcome) pairs
3. Train MLP to predict game outcome from state features
4. Export trained weights to value_weights.npz

Usage:
  python -m MaxBeater.train_value_model --games 1000 --epochs 50 --batch-size 256
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Tuple, Dict
from collections import deque
import numpy as np

# PyTorch imports (only available during training, not runtime)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch not available. This script is for offline training only.")
    sys.exit(1)

# Import game engine (same style as runtime agent)
from game.board import Board
from game.game_map import GameMap
from game.enums import Direction, MoveType, Result

# Import our runtime components (package-relative imports)
from .belief import TrapdoorBelief
from .features import extract_scalar_features, encode_board_tensor


class ValueNet(nn.Module):
    """
    Neural network for value estimation.
    
    Architecture matches value_model_runtime.py:
      Input: 1050 dims (14*8*8 + 26)
      Hidden1: 256 units, ReLU
      Hidden2: 128 units, ReLU
      Output: 1 unit, tanh
    """
    
    def __init__(self, input_dim: int = 1050, hidden1: int = 256, hidden2: int = 128):
        super(ValueNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class GameDataset(Dataset):
    """PyTorch dataset for training data."""
    
    def __init__(self, states: List[np.ndarray], values: List[float]):
        self.states = torch.FloatTensor(np.array(states))
        self.values = torch.FloatTensor(np.array(values)).unsqueeze(1)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.values[idx]


def play_game_for_training(use_mcts: bool = False) -> List[Tuple[np.ndarray, float]]:
    """
    Play one game using random or greedy play to generate training data.
    
    Args:
        use_mcts: If True, use simple MCTS-like search; else use random play
        
    Returns:
        List of (state_features, outcome_value) tuples
    """
    # Initialize game
    game_map = GameMap()
    game_board = Board(game_map, time_to_play=360.0, build_history=False)
    
    # Set up chickens at random spawn positions
    spawns = [(0, np.random.randint(1, 7)), (7, np.random.randint(1, 7))]
    game_board.chicken_player.start(spawns[0], 0)
    game_board.chicken_enemy.start(spawns[1], 1)
    
    # Initialize belief tracker
    trap_belief = TrapdoorBelief(board_size=8)
    
    # Dummy sensor data (TODO: integrate with real trapdoor manager)
    dummy_sensors = [(False, False), (False, False)]
    
    # Collect (state, outcome) pairs
    states_features = []
    
    # Play game
    turn_count = 0
    max_turns = 80  # 40 per player
    
    while not game_board.is_game_over() and turn_count < max_turns:
        # Get valid moves
        valid_moves = game_board.get_valid_moves(enemy=False)
        
        if not valid_moves:
            break
        
        # Save current state features
        board_tensor = encode_board_tensor(game_board, trap_belief)
        scalars = extract_scalar_features(game_board, trap_belief)
        state_features = np.concatenate([board_tensor.flatten(), scalars])
        states_features.append(state_features)
        
        # Choose move (random or greedy)
        if use_mcts:
            # Simple greedy: pick move that maximizes immediate egg differential
            best_move = valid_moves[0]
            best_score = -999999
            for move in valid_moves:
                child_board = game_board.forecast_move(move[0], move[1])
                if child_board is not None:
                    score = child_board.chicken_player.get_eggs_laid() - child_board.chicken_enemy.get_eggs_laid()
                    if score > best_score:
                        best_score = score
                        best_move = move
            move = best_move
        else:
            # Random
            move = valid_moves[np.random.randint(len(valid_moves))]
        
        # Apply move
        game_board.apply_move(move[0], move[1], check_ok=True)
        
        # Switch perspective
        game_board.reverse_perspective()
        
        turn_count += 1
    
    # Determine outcome
    winner = game_board.get_winner()
    my_eggs = game_board.chicken_player.get_eggs_laid()
    enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
    
    # Assign values based on outcome
    # Use egg differential normalized to [-1, 1]
    egg_diff = my_eggs - enemy_eggs
    outcome_value = np.tanh(egg_diff / 10.0)  # Normalize
    
    # Assign alternating values (perspective flips each turn)
    training_pairs = []
    for i, state in enumerate(states_features):
        # Alternate sign for each turn
        if i % 2 == 0:
            value = outcome_value
        else:
            value = -outcome_value
        training_pairs.append((state, value))
    
    return training_pairs


def collect_training_data(num_games: int, use_mcts: bool = False) -> Tuple[List[np.ndarray], List[float]]:
    """
    Collect training data from self-play games.
    
    Args:
        num_games: Number of games to play
        use_mcts: Use greedy search instead of random
        
    Returns:
        (states, values) tuple
    """
    all_states = []
    all_values = []
    
    print(f"Collecting training data from {num_games} games...")
    
    for game_num in range(num_games):
        if (game_num + 1) % 10 == 0:
            print(f"  Played {game_num + 1}/{num_games} games, collected {len(all_states)} samples")
        
        try:
            game_data = play_game_for_training(use_mcts=use_mcts)
            
            for state, value in game_data:
                all_states.append(state)
                all_values.append(value)
        except Exception as e:
            print(f"  Warning: Game {game_num} failed: {e}")
            continue
    
    print(f"Collected {len(all_states)} training samples from {num_games} games")
    return all_states, all_values


def train_model(
    states: List[np.ndarray],
    values: List[float],
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = 'cuda'
) -> ValueNet:
    """
    Train the value network.
    
    Args:
        states: List of state feature vectors
        values: List of target values
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
        
    Returns:
        Trained model
    """
    # Create dataset and dataloader
    dataset = GameDataset(states, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = len(states[0])
    model = ValueNet(input_dim=input_dim)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining model on {device}...")
    print(f"  Dataset: {len(states)} samples")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Architecture: {input_dim} -> 256 -> 128 -> 1")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_states, batch_values in dataloader:
            batch_states = batch_states.to(device)
            batch_values = batch_values.to(device)
            
            # Forward pass
            predictions = model(batch_states)
            loss = criterion(predictions, batch_values)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    print("Training complete!")
    return model


def export_weights(model: ValueNet, output_path: str):
    """
    Export trained PyTorch model weights to NumPy .npz format.
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save weights file
    """
    model.eval()
    
    # Extract weights and biases as NumPy arrays
    W1 = model.fc1.weight.detach().cpu().numpy()
    b1 = model.fc1.bias.detach().cpu().numpy()
    
    W2 = model.fc2.weight.detach().cpu().numpy()
    b2 = model.fc2.bias.detach().cpu().numpy()
    
    W3 = model.fc3.weight.detach().cpu().numpy()
    b3 = model.fc3.bias.detach().cpu().numpy()
    
    # Save to .npz file
    np.savez(output_path, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    
    print(f"\nWeights exported to {output_path}")
    print(f"  W1: {W1.shape}, b1: {b1.shape}")
    print(f"  W2: {W2.shape}, b2: {b2.shape}")
    print(f"  W3: {W3.shape}, b3: {b3.shape}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train value function for MaxBeater')
    parser.add_argument('--games', type=int, default=500, help='Number of self-play games')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use-mcts', action='store_true', help='Use greedy search instead of random')
    parser.add_argument('--output', type=str, default='value_weights.npz', help='Output weights file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("MaxBeater Value Function Training")
    print("="*60)
    
    # Step 1: Collect training data
    states, values = collect_training_data(args.games, use_mcts=args.use_mcts)
    
    if len(states) == 0:
        print("ERROR: No training data collected!")
        return
    
    # Step 2: Train model
    model = train_model(
        states,
        values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Step 3: Export weights
    export_weights(model, args.output)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Weights saved to: {args.output}")
    print("Copy this file to the MaxBeater directory for runtime use.")
    print("="*60)


if __name__ == "__main__":
    main()

