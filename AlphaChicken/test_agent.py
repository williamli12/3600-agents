"""
test_agent.py - Quick verification script for AlphaChicken

Run this to verify the agent can be imported and initialized correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import board as game_board
from game.game_map import GameMap
from AlphaChicken.agent import PlayerAgent


def test_agent_initialization():
    """Test that the agent can be initialized."""
    print("Testing AlphaChicken initialization...")
    
    # Create a mock game board
    game_map = GameMap()
    test_board = game_board.Board(game_map, time_to_play=360)
    
    # Initialize chickens
    test_board.chicken_player.start((0, 4), 0)
    test_board.chicken_enemy.start((7, 4), 1)
    
    # Create time_left function
    def time_left():
        return 360.0
    
    # Initialize agent
    try:
        agent = PlayerAgent(test_board, time_left, seed=42)
        print("✓ Agent initialized successfully!")
        return True
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_play():
    """Test that the agent can make a move."""
    print("\nTesting AlphaChicken play...")
    
    # Create a mock game board
    game_map = GameMap()
    test_board = game_board.Board(game_map, time_to_play=360)
    
    # Initialize chickens
    test_board.chicken_player.start((0, 4), 0)
    test_board.chicken_enemy.start((7, 4), 1)
    
    # Create time_left function
    def time_left():
        return 360.0
    
    # Initialize agent
    agent = PlayerAgent(test_board, time_left, seed=42)
    
    # Mock sensor data (no trapdoors detected)
    sensor_data = [(False, False), (False, False)]
    
    # Get a move
    try:
        move = agent.play(test_board, sensor_data, time_left)
        direction, move_type = move
        print(f"✓ Agent returned move: {move}")
        
        # Verify it's a valid move
        valid_moves = test_board.get_valid_moves()
        if move in valid_moves:
            print("✓ Move is valid!")
            return True
        else:
            print(f"✗ Move {move} not in valid moves: {valid_moves}")
            return False
    except Exception as e:
        print(f"✗ Agent play failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_architecture():
    """Test that the model can be created and performs forward pass."""
    print("\nTesting AlphaChickenNet architecture...")
    
    try:
        import torch
        from AlphaChicken.model import AlphaChickenNet, board_to_tensor
        from game import board as game_board
        from game.game_map import GameMap
        
        # Create model
        model = AlphaChickenNet(num_channels=128, num_res_blocks=4)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create test input
        game_map = GameMap()
        test_board = game_board.Board(game_map, time_to_play=360)
        test_board.chicken_player.start((0, 4), 0)
        test_board.chicken_enemy.start((7, 4), 1)
        
        state_tensor = board_to_tensor(test_board, trap_belief=None)
        print(f"✓ State tensor shape: {state_tensor.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            policy_logits, value = model(state_tensor)
        
        print(f"✓ Policy output shape: {policy_logits.shape}")
        print(f"✓ Value output shape: {value.shape}")
        print(f"✓ Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("AlphaChicken Verification Tests")
    print("="*60)
    
    results = []
    
    # Test 1: Model architecture
    results.append(("Model Architecture", test_model_architecture()))
    
    # Test 2: Agent initialization
    results.append(("Agent Initialization", test_agent_initialization()))
    
    # Test 3: Agent play
    results.append(("Agent Play", test_agent_play()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*60)
    if all_passed:
        print("All tests passed! AlphaChicken is ready to use. 🎉")
    else:
        print("Some tests failed. Please check the errors above.")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


