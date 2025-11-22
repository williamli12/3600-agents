"""
train_pace.py - Training loop for AlphaChicken

Implements self-play training with policy and value learning.
Run this on a cluster to train the network.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from collections import deque

from game import board as game_board_module
from game import enums
from game.game_map import GameMap
from game.trapdoor_manager import TrapdoorManager

from model import AlphaChickenNet, board_to_tensor, index_to_move, move_to_index
from mcts import MCTS
from agent import TrapdoorBelief


class ReplayBuffer:
    """Store training data from self-play games."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, policy_target, value_target):
        """Add a training example."""
        self.buffer.append((state, policy_target, value_target))
    
    def sample(self, batch_size: int):
        """Sample a batch of training data."""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.stack([item[0] for item in batch])
        policies = torch.stack([item[1] for item in batch])
        values = torch.stack([item[2] for item in batch])
        
        return states, policies, values
    
    def __len__(self):
        return len(self.buffer)


def play_game(model: AlphaChickenNet, num_simulations: int = 50, device: str = 'cpu'):
    """
    Play one self-play game and collect training data.
    
    Args:
        model: Neural network to use for MCTS
        num_simulations: Number of MCTS simulations per move
        device: PyTorch device
        
    Returns:
        List of (state_tensor, policy_target, value_target) tuples
    """
    # Initialize game
    game_map = GameMap()
    trapdoor_manager = TrapdoorManager(game_map)
    board = game_board_module.Board(game_map, time_to_play=360, build_history=False)
    
    # Set up spawns and trapdoors
    spawns = trapdoor_manager.choose_spawns()
    trapdoor_locations = trapdoor_manager.choose_trapdoors()
    board.chicken_player.start(spawns[0], 0)
    board.chicken_enemy.start(spawns[1], 1)
    
    # Initialize belief trackers (one per player)
    trap_belief_a = TrapdoorBelief(board_size=8)
    trap_belief_b = TrapdoorBelief(board_size=8)
    
    # Initialize MCTS
    mcts_a = MCTS(model, trap_belief_a, c_puct=1.5, device=device)
    mcts_b = MCTS(model, trap_belief_b, c_puct=1.5, device=device)
    
    # Storage for training data
    game_history = []  # List of (state, policy, player_id)
    
    move_count = 0
    max_moves = 80  # 40 turns per player
    
    while not board.is_game_over() and move_count < max_moves:
        # Determine whose turn it is
        is_player_a = board.is_as_turn
        current_trap_belief = trap_belief_a if is_player_a else trap_belief_b
        current_mcts = mcts_a if is_player_a else mcts_b
        
        # Update trapdoor beliefs with sensor data
        player_location = board.chicken_player.get_location()
        sensor_data = trapdoor_manager.sample_trapdoors(player_location)
        current_trap_belief.update(board, sensor_data)
        
        # Get state representation
        state_tensor = board_to_tensor(board, current_trap_belief)
        
        # Run MCTS
        root = current_mcts.search(board, num_simulations=num_simulations)
        
        # Get policy target (visit count distribution)
        temperature = 1.0 if move_count < 30 else 0.0  # Exploration early, exploitation late
        action_probs = current_mcts.get_action_probs(root, temperature=temperature)
        
        # Store state and policy for training
        policy_tensor = torch.from_numpy(action_probs)
        player_id = 0 if is_player_a else 1
        game_history.append((state_tensor.squeeze(0), policy_tensor, player_id))
        
        # Choose move
        if temperature == 0:
            move = current_mcts.get_best_move(root)
        else:
            # Sample from distribution
            valid_moves = board.get_valid_moves()
            valid_indices = [move_to_index(d, m) for d, m in valid_moves]
            
            if len(valid_indices) > 0:
                probs = action_probs[valid_indices]
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    chosen_idx = np.random.choice(valid_indices, p=probs)
                    move = index_to_move(chosen_idx)
                else:
                    move = valid_moves[0]
            else:
                move = (enums.Direction.UP, enums.MoveType.PLAIN)
        
        # Apply move
        direction, move_type = move
        valid = board.apply_move(direction, move_type, timer=0.0)
        
        if not valid:
            # Invalid move - penalize and end game
            board.set_winner(enums.Result.ENEMY, enums.WinReason.INVALID_TURN)
            break
        
        # Check for trapdoor
        new_location = board.chicken_player.get_location()
        if trapdoor_manager.is_trapdoor(new_location):
            board.chicken_player.reset_location()
            board.chicken_enemy.increment_eggs_laid(-1 * board.game_map.TRAPDOOR_PENALTY)
            board.found_trapdoors.add(new_location)
        
        # Flip perspective for next player
        if not board.is_game_over():
            board.reverse_perspective()
        
        move_count += 1
    
    # Game over - assign value targets
    winner = board.get_winner()
    
    # Determine winner from original perspective
    if winner == enums.Result.PLAYER:
        # Current player won (after final reverse_perspective)
        # Need to track who actually won
        final_is_a = board.is_as_turn
        if final_is_a:
            winner_id = 0  # Player A won
        else:
            winner_id = 1  # Player B won
    elif winner == enums.Result.ENEMY:
        final_is_a = board.is_as_turn
        if final_is_a:
            winner_id = 1  # Player B won
        else:
            winner_id = 0  # Player A won
    else:  # TIE
        winner_id = -1
    
    # Create training examples
    training_data = []
    for state, policy, player_id in game_history:
        if winner_id == -1:
            value_target = 0.0
        elif winner_id == player_id:
            value_target = 1.0
        else:
            value_target = -1.0
        
        value_tensor = torch.tensor([value_target], dtype=torch.float32)
        training_data.append((state, policy, value_tensor))
    
    return training_data


def train_model(model: AlphaChickenNet, optimizer: optim.Optimizer, 
                replay_buffer: ReplayBuffer, batch_size: int = 256, device: str = 'cpu'):
    """
    Train the model on a batch from the replay buffer.
    
    Args:
        model: Neural network
        optimizer: Optimizer
        replay_buffer: Replay buffer with training data
        batch_size: Batch size
        device: PyTorch device
        
    Returns:
        (policy_loss, value_loss, total_loss)
    """
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0, 0.0
    
    model.train()
    
    # Sample batch
    states, policy_targets, value_targets = replay_buffer.sample(batch_size)
    states = states.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device)
    
    # Forward pass
    policy_logits, value_preds = model(states)
    
    # Compute losses
    policy_loss = -torch.sum(policy_targets * policy_logits) / batch_size
    value_loss = nn.MSELoss()(value_preds, value_targets)
    
    total_loss = policy_loss + value_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    model.eval()
    
    return policy_loss.item(), value_loss.item(), total_loss.item()


def main():
    """Main training loop."""
    # Hyperparameters
    num_epochs = 100
    games_per_epoch = 50
    num_simulations = 50
    batch_size = 256
    learning_rate = 0.001
    replay_buffer_size = 10000
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    model = AlphaChickenNet(num_channels=128, num_res_blocks=4)
    model.to(device)
    model.eval()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Self-play phase
        print(f"Generating {games_per_epoch} self-play games...")
        for game_idx in range(games_per_epoch):
            try:
                game_data = play_game(model, num_simulations=num_simulations, device=device)
                
                # Add to replay buffer
                for state, policy, value in game_data:
                    replay_buffer.add(state, policy, value)
                
                if (game_idx + 1) % 10 == 0:
                    print(f"  Game {game_idx + 1}/{games_per_epoch} complete, buffer size: {len(replay_buffer)}")
            except Exception as e:
                print(f"  Game {game_idx + 1} failed: {e}")
                continue
        
        # Training phase
        print(f"\nTraining on {len(replay_buffer)} examples...")
        num_train_steps = max(len(replay_buffer) // batch_size, 1) * 5  # 5 epochs per self-play round
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        
        for step in range(num_train_steps):
            policy_loss, value_loss, loss = train_model(
                model, optimizer, replay_buffer, batch_size=batch_size, device=device
            )
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += loss
        
        avg_policy_loss = total_policy_loss / num_train_steps
        avg_value_loss = total_value_loss / num_train_steps
        avg_total_loss = total_loss / num_train_steps
        
        epoch_time = time.time() - epoch_start
        
        # Log results
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Value Loss:  {avg_value_loss:.4f}")
        print(f"  Total Loss:  {avg_total_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        checkpoint_path = f'checkpoints/model_epoch_{epoch + 1}.pt'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save as best model
        best_model_path = 'best_model.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best_model_path)
        print(f"  Updated best model: {best_model_path}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

