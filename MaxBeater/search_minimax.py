"""
search_minimax.py - Alpha-beta Minimax search for MaxBeater
"""

import time
import math
from typing import Tuple, Optional, List, Dict
from game import board, enums
from .belief import TrapdoorBelief
from .evaluator import Evaluator

# Constants
MAX_DEPTH = 9
infinity = float('inf')

class MinimaxSearch:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        # Transposition table: key -> (depth, score, flag)
        # flag: 0=exact, 1=lowerbound (alpha), 2=upperbound (beta)
        self.tt = {}
        self.nodes_explored = 0
        self.start_time = 0.0
        self.time_budget = 0.0
        self.timed_out = False

    def choose_best_move(
        self, 
        game_board: board.Board, 
        time_left_fn, 
        belief: TrapdoorBelief, 
        visit_counts: Dict[Tuple[int, int], int]
    ) -> Tuple[enums.Direction, enums.MoveType]:
        
        self.start_time = time.time()
        time_left = time_left_fn()
        
        # Time management
        # allocate roughly (time_left / turns_left) * factor
        # limit max time per move to avoid timeout
        turns = game_board.turns_left_player
        if turns < 1: turns = 1
        
        # Budget: more time early/mid game, less late game? 
        # Actually constant time per move is safer.
        self.time_budget = min(time_left / turns * 1.5, 2.0) 
        if time_left < 5.0:
            self.time_budget = 0.2 # Panic mode
        elif time_left < 15.0:
            self.time_budget = 0.5

        self.nodes_explored = 0
        self.timed_out = False
        
        # Iterative deepening?
        # The prompt asks for fixed depth 9, but handling time limit suggests ID might be safer.
        # However, let's try to hit depth 9 directly, and abort if time runs out.
        # To ensure we return *something*, we should probably capture the best move from a shallower search first.
        # Let's just run the search at MAX_DEPTH and have it check time.
        
        best_move = None
        best_val = -infinity
        
        # Root move generation & ordering
        moves = self._get_ordered_moves(game_board, is_max=True)
        if not moves:
            return (enums.Direction.UP, enums.MoveType.PLAIN) # Fallback

        alpha = -infinity
        beta = infinity
        
        best_move = moves[0][0] # Default to first ordered move

        for move, move_type in moves:
            # Apply move
            # We need a copy for the recursive step
            next_board = game_board.forecast_move(move, move_type, check_ok=False)
            if next_board is None: continue
            
            # Update visit counts for the recursion (add current pos)
            # But wait, the move *leads* to a new pos. The penalty is for visiting squares *in the search path*.
            # The prompt says: "revisit_penalty = sum(visit_counts[r, c] for current player's path)"
            # So we should pass a COPY of visit_counts with the new location incremented.
            
            # My location after move:
            # forecast_move updates the board.
            # So next_board.chicken_player.get_location() is the new location.
            new_pos = next_board.chicken_player.get_location()
            new_visits = visit_counts.copy()
            new_visits[new_pos] = new_visits.get(new_pos, 0) + 1
            
            # Call minimax
            val = self._minimax(
                next_board, 
                1, 
                alpha, 
                beta, 
                False, # minimizing player (enemy)
                belief, 
                new_visits
            )
            
            if self.timed_out:
                break
            
            print(f"Move {move.name} {move_type.name}: {val}")
            
            if val > best_val:
                best_val = val
                best_move = (move, move_type)
            
            alpha = max(alpha, best_val)
        
        print(f"Search finished. Nodes: {self.nodes_explored}, Best: {best_val}")
        return best_move

    def _minimax(
        self, 
        board_state: board.Board, 
        depth: int, 
        alpha: float, 
        beta: float, 
        is_max: bool, 
        belief: TrapdoorBelief, 
        visit_counts: Dict[Tuple[int, int], int]
    ) -> float:
        
        # 1. Time check (periodically)
        if self.nodes_explored % 1000 == 0:
            if time.time() - self.start_time > self.time_budget:
                self.timed_out = True
                return 0.0 # Return dummy value, will be discarded by root
        
        self.nodes_explored += 1
        
        # 2. TT Lookup
        tt_key = self._make_tt_key(board_state, belief)
        if tt_key in self.tt:
            entry_depth, entry_val, entry_flag = self.tt[tt_key]
            # We can use this entry if it was searched to at least the same depth
            # Remaining depth = MAX_DEPTH - current_depth
            # Stored depth should be >= remaining depth?
            # Let's interpret 'depth' in TT as 'depth remaining' or 'depth searched'.
            # The prompt says: "store (depth, score), and only use entry if stored depth >= current depth"
            # Wait, usually 'depth' in TT refers to 'depth of subtree searched'.
            # Here, 'depth' param increases 0..9.
            # So remaining depth is MAX_DEPTH - depth.
            remaining = MAX_DEPTH - depth
            if entry_depth >= remaining:
                if entry_flag == 0: # Exact
                    return entry_val
                elif entry_flag == 1: # Lowerbound
                    alpha = max(alpha, entry_val)
                elif entry_flag == 2: # Upperbound
                    beta = min(beta, entry_val)
                
                if alpha >= beta:
                    return entry_val

        # 3. Terminal / Leaf
        if depth >= MAX_DEPTH or board_state.is_game_over():
            # Evaluate
            # Note: evaluate_state is from perspective of chicken_player (us)
            # If is_max is False, it means it's enemy's turn to move in the recursion.
            # But the evaluation function should always return score for US.
            return self.evaluator.evaluate_state(board_state, belief, visit_counts)

        # 4. Expansion
        # Get ordered moves
        # Note: board_state handles 'current player' automatically based on whose turn it is.
        # But we need to know if 'current player' in board_state matches our 'is_max'.
        # MaxBeater is always player A (or B) and is maximizing.
        # In Minimax, if is_max=True, it's MaxBeater's turn.
        # If is_max=False, it's Enemy's turn.
        # board_state tracks turns.
        
        moves = self._get_ordered_moves(board_state, is_max)
        
        best_val = -infinity if is_max else infinity
        tt_flag = 0 # Exact
        
        for move, move_type in moves:
            # Check timeout again to abort quickly
            if self.timed_out:
                return best_val
                
            # Apply move
            # Use copy=True implicit in forecast_move
            next_board = board_state.forecast_move(move, move_type, check_ok=False)
            if next_board is None: continue
            
            # Update visit counts if maximizing player
            new_visits = visit_counts
            if is_max:
                new_pos = next_board.chicken_player.get_location()
                new_visits = visit_counts.copy()
                new_visits[new_pos] = new_visits.get(new_pos, 0) + 1
            
            val = self._minimax(next_board, depth + 1, alpha, beta, not is_max, belief, new_visits)
            
            if is_max:
                if val > best_val:
                    best_val = val
                alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_val = val
                beta = min(beta, best_val)
            
            if alpha >= beta:
                tt_flag = 1 if is_max else 2 # Cutoff
                break
        
        # 5. Store in TT
        remaining = MAX_DEPTH - depth
        if not self.timed_out:
            self.tt[tt_key] = (remaining, best_val, tt_flag)
            
        return best_val

    def _get_ordered_moves(self, game_board: board.Board, is_max: bool) -> List[Tuple[enums.Direction, enums.MoveType]]:
        """
        Generate and order moves.
        Filtering logic:
        - If EGG possible in a direction, ignore PLAIN/TURD for that direction.
        - No TURD on edges.
        Ordering: EGG > PLAIN (good) > TURD
        """
        valid_moves = game_board.get_valid_moves(enemy=False) # 'False' means current turn player
        if not valid_moves:
            return []
            
        # Filtering
        # Group by direction
        moves_by_dir = {}
        for d, mtype in valid_moves:
            if d not in moves_by_dir: moves_by_dir[d] = []
            moves_by_dir[d].append(mtype)
        
        filtered_moves = []
        
        # Determine if we should suppress turds based on egg count (heuristic tweak)
        # "discourage TURD before we have laid at least a threshold of eggs (e.g., 4–6)"
        # This is better handled in ordering scores than hard filtering, but we can penalize.
        
        my_loc = game_board.chicken_player.get_location()
        
        for d, types in moves_by_dir.items():
            has_egg = enums.MoveType.EGG in types
            
            for mtype in types:
                # Rule: If egg is possible, skip others?
                if has_egg and mtype != enums.MoveType.EGG:
                     # "if you can lay an egg on the destination, do it"
                     continue
                
                # Rule: Forbid TURD on edge squares
                if mtype == enums.MoveType.TURD:
                    # Need to check destination
                    # Turd is dropped AT CURRENT LOCATION usually? 
                    # game/chicken.py: drop_turd() -> self.turds_left -= 1. Returns loc.
                    # But apply_move in board.py: 
                    # if TURD: self.turds_player.add(my_loc)
                    # new_loc = apply_dir(dir)
                    # So turd is placed at start loc, then we move.
                    # "Globally forbid placing TURDs on edge squares"
                    # So check if current loc is edge.
                    x, y = my_loc
                    if x == 0 or x == 7 or y == 0 or y == 7:
                        continue
                
                filtered_moves.append((d, mtype))
        
        # Ordering
        def score_move(move_tuple):
            d, mtype = move_tuple
            # Base scores
            if mtype == enums.MoveType.EGG:
                score = 100
            elif mtype == enums.MoveType.PLAIN:
                score = 50
            else: # TURD
                score = 10
            
            # Heuristic adjustments
            # E.g. move towards center?
            # Get dest
            dest = enums.loc_after_direction(my_loc, d)
            dist_center = abs(dest[0] - 3.5) + abs(dest[1] - 3.5)
            score -= dist_center * 2 # Closer to center is better
            
            return score
            
        filtered_moves.sort(key=score_move, reverse=True)
        return filtered_moves

    def _make_tt_key(self, game_board: board.Board, belief: TrapdoorBelief):
        # Hashable state representation
        # Player turn is part of state (captured by board state mostly, but explicit is good)
        # We need positions, eggs, turds.
        # board_copy = game_board.get_copy() # Too slow
        
        # Assuming board internals:
        p_loc = game_board.chicken_player.get_location()
        e_loc = game_board.chicken_enemy.get_location()
        
        # Sets to sorted tuples
        p_eggs = tuple(sorted(list(game_board.eggs_player)))
        e_eggs = tuple(sorted(list(game_board.eggs_enemy)))
        p_turds = tuple(sorted(list(game_board.turds_player)))
        e_turds = tuple(sorted(list(game_board.turds_enemy)))
        
        # Belief snapshot
        b_snap = belief.snapshot()
        
        return (
            game_board.is_as_turn, 
            p_loc, e_loc, 
            p_eggs, e_eggs, 
            p_turds, e_turds,
            b_snap
        )

