"""
agent.py - Main agent orchestrator for MaxBeater

MaxBeater is a strong AI agent combining:
- Bayesian trapdoor belief tracking
- Hand-crafted heuristics + learned value function
- MCTS search with UCT
- Fast fallback search for time pressure

Architecture:
  TrapdoorBelief -> updates beliefs from sensor data
  Evaluator -> combines heuristic + value model
  MCTSSearch -> main search engine
  FallbackSearch -> fast search when time is low

The agent maintains the same external interface required by the game engine:
  __init__(board, time_left)
  play(board, sensor_data, time_left) -> (Direction, MoveType)
"""

from __future__ import annotations
from collections.abc import Callable
from typing import List, Tuple, Optional
import numpy as np
from game import board, enums

# Import our components
from .belief import TrapdoorBelief
from .evaluator import Evaluator
from .search_mcts import MCTSSearch
from .search_fallback import FallbackSearch


class PlayerAgent:
    """
    MaxBeater - Strong agent with MCTS + learned evaluation.
    
    This class is instantiated by the game engine and must maintain
    the expected interface (__init__ and play methods with specific signatures).
    """
    
    def __init__(self, game_board: board.Board, time_left: Callable):
        """
        Initialize the agent and all components.
        
        Args:
            game_board: Initial board state
            time_left: Function returning remaining time in seconds
        """
        # Random number generator for reproducibility and tie-breaking
        seed = 42
        self.rng = np.random.default_rng(seed)
        
        print("[MaxBeater] Initializing agent...")
        
        # === Component 1: Trapdoor Belief Tracker ===
        # Maintains Bayesian beliefs over trapdoor locations
        self.trap_belief = TrapdoorBelief(board_size=8)
        print("[MaxBeater] Trapdoor belief system initialized")
        
        # === Component 2: Evaluator ===
        # Combines heuristic evaluation with learned value function
        self.evaluator = Evaluator(self.trap_belief)
        print("[MaxBeater] Evaluator initialized (heuristic + value model)")
        
        # === Component 3: Search Engines ===
        # MCTS for main search
        self.mcts = MCTSSearch(self.evaluator, self.trap_belief, self.rng)
        print("[MaxBeater] MCTS search engine initialized")
        
        # Fallback for low-time situations
        self.fallback = FallbackSearch(self.evaluator, self.trap_belief, self.rng)
        print("[MaxBeater] Fallback search engine initialized")
        
        # === Timing parameters ===
        self.low_time_threshold = 15.0  # Switch to fallback below this
        self.critical_time_threshold = 5.0  # Emergency mode
        
        print("[MaxBeater] Initialization complete!")
    
    def play(
        self,
        game_board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Choose and return the best move for the current turn.
        
        Args:
            game_board: Current board state
            sensor_data: List of (heard, felt) tuples for each trapdoor
            time_left: Function returning remaining time in seconds
            
        Returns:
            (Direction, MoveType) tuple representing the chosen move
        """
        # === Step 1: Update trapdoor beliefs ===
        self.trap_belief.update(game_board, sensor_data)
        
        # === Step 2: Get valid moves ===
        valid_moves = game_board.get_valid_moves()
        
        # === Step 3: Check time and decide search strategy ===
        remaining_time = time_left()
        turns_left = game_board.turns_left_player
        
        # Log current state
        my_pos = game_board.chicken_player.get_location()
        my_eggs = game_board.chicken_player.get_eggs_laid()
        enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
        
        print(f"\n[MaxBeater] Turn {game_board.turn_count}")
        print(f"  Position: {my_pos}, Eggs: {my_eggs} vs {enemy_eggs}")
        print(f"  Time: {remaining_time:.1f}s, Turns left: {turns_left}")
        print(f"  Sensors: {sensor_data}")
        
        # ------------------------------------------------------------------
        # Early-egg rule:
        # If no eggs have been laid by either player yet and it's still
        # early in the game, and we have at least one EGG move available,
        # choose an egg move directly instead of running search.
        # ------------------------------------------------------------------
        total_eggs = my_eggs + enemy_eggs
        
        if total_eggs == 0 and turns_left > 30:
            egg_moves = [
                m for m in valid_moves
                if m[1] == enums.MoveType.EGG
            ]
            
            if egg_moves:
                best_move = egg_moves[0]
                best_score = -float("inf")
                
                for move in egg_moves:
                    # Reuse fallback's apply_move to simulate the move
                    child_board = self.fallback._apply_move(game_board, move)
                    if child_board is None:
                        continue
                    
                    # Evaluate resulting position from our perspective
                    score = self.evaluator.quick_evaluate(child_board)
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                
                print("[MaxBeater] [RULE] Early-egg rule triggered, choosing egg move:", best_move)
                return best_move
        
        # ------------------------------------------------------------------
        # Mid/Late-game egg rule:
        # If we are behind in eggs OR in late game and we have an EGG move,
        # pick the best EGG move directly.
        # ------------------------------------------------------------------
        # Recompute in case we add more rules later
        total_eggs = my_eggs + enemy_eggs
        
        # Conditions to trigger this rule:
        # - we have fewer eggs than the enemy, OR
        # - we are in late game (few turns left)
        should_force_egg = (my_eggs < enemy_eggs) or (turns_left <= 8)
        
        if should_force_egg:
            egg_moves = [
                m for m in valid_moves
                if m[1] == enums.MoveType.EGG
            ]
            
            if egg_moves:
                best_move = egg_moves[0]
                best_score = -float("inf")
                
                for move in egg_moves:
                    child_board = self.fallback._apply_move(game_board, move)
                    if child_board is None:
                        continue
                    
                    score = self.evaluator.quick_evaluate(child_board)
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                
                print("[MaxBeater] [RULE] Mid/Late-game egg rule triggered, choosing egg move:", best_move)
                return best_move
        
        # === Step 4: Choose move using appropriate search ===
        move = None
        
        if remaining_time < self.critical_time_threshold:
            # CRITICAL TIME: Use fastest possible search
            print(f"  [CRITICAL] Using greedy fallback")
            move = self.fallback.choose_move(game_board, time_left)
            
        elif remaining_time < self.low_time_threshold:
            # LOW TIME: Use fallback search (1-2 ply)
            print(f"  [LOW TIME] Using fallback search")
            move = self.fallback.choose_move(game_board, time_left)
            
        else:
            # NORMAL TIME: Use full MCTS
            print(f"  [NORMAL] Using MCTS")
            move = self.mcts.choose_move(game_board, time_left, turns_left)
        
        # === Step 5: Safety checks ===
        if not valid_moves:
            # No valid moves (shouldn't happen, but handle gracefully)
            print("[MaxBeater] WARNING: No valid moves available!")
            # Return a default move (will be rejected by engine, but better than crashing)
            return (enums.Direction.UP, enums.MoveType.PLAIN)
        
        if move is None or move not in valid_moves:
            # Search failed or returned invalid move: fallback to random valid
            print(f"[MaxBeater] WARNING: Invalid move {move}, choosing random valid move")
            move = valid_moves[self.rng.integers(0, len(valid_moves))]
        
        print(f"  Selected move: {move}")
        
        return move


# Optional: Helper function for testing/debugging
def test_agent():
    """
    Simple test function to verify agent can be instantiated.
    Not called during actual gameplay.
    """
    from game.game_map import GameMap
    from game.board import Board
    
    # Create dummy board
    game_map = GameMap()
    test_board = Board(game_map, time_to_play=360.0)
    
    # Initialize chickens
    test_board.chicken_player.start((0, 3), 0)
    test_board.chicken_enemy.start((7, 3), 1)
    
    # Create agent
    def dummy_time_left():
        return 180.0
    
    agent = PlayerAgent(test_board, dummy_time_left)
    
    # Try one move
    sensor_data = [(False, False), (False, False)]
    move = agent.play(test_board, sensor_data, dummy_time_left)
    
    print(f"Test agent returned move: {move}")
    print("Agent test successful!")


if __name__ == "__main__":
    # Run test if executed directly
    test_agent()
