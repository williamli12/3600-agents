import time
import math
import random
from typing import Optional, Tuple, List, Callable
from game import board, enums

from .transposition import TranspositionTable, TTFlag
from .evaluation import Evaluator

class SearchEngine:
    def __init__(self, evaluator: Evaluator, tt: TranspositionTable):
        self.evaluator = evaluator
        self.tt = tt
        self.max_depth = 20
        self.time_buffer = 0.5 # Seconds to leave as buffer
        self.nodes_visited = 0
        
    def search(self, game_board: board.Board, time_left: Callable[[], float]) -> Tuple[enums.Direction, enums.MoveType]:
        """
        Iterative deepening search.
        """
        self.nodes_visited = 0
        start_time = time.time()
        best_move = None
        
        valid_moves = game_board.get_valid_moves()
        if not valid_moves:
            return None # Should not happen
            
        if len(valid_moves) == 1:
            return valid_moves[0]
            
        # Time Management
        total_time = time_left()
        # Allocate time per move: e.g. remaining / 20, but at least 0.5s
        # But game has fixed turns (40).
        turns_remaining = max(1, game_board.turns_left_player)
        time_limit = (total_time - self.time_buffer) / (turns_remaining * 0.8) # Dynamic allocation
        time_limit = min(time_limit, 5.0) # Cap at 5s
        time_limit = max(time_limit, 0.2) # Min 0.2s
        
        # If extremely low time, just move fast
        if total_time < 1.0:
            time_limit = 0.1
            
        end_time = start_time + time_limit
        
        alpha = -math.inf
        beta = math.inf
        
        current_depth = 1
        
        try:
            while current_depth <= self.max_depth:
                # Check time before starting new depth
                if time.time() > end_time:
                    break
                    
                # PVS Search at Root
                # Root is always PV node
                move, score = self._search_root(game_board, current_depth, alpha, beta, end_time)
                
                if move:
                    best_move = move
                    # Aspiration Window adjustments?
                    # For now, full window at root is fine.
                
                # Prepare for next depth
                current_depth += 1
                
        except TimeoutError:
            pass # Time out, return best so far
            
        return best_move if best_move else random.choice(valid_moves)

    def _search_root(self, game_board, depth, alpha, beta, end_time):
        """
        Root search handling for PVS/Iterative Deepening.
        """
        valid_moves = game_board.get_valid_moves()
        if not valid_moves:
            return None, -math.inf

        # Move Ordering
        ordered_moves = self._order_moves(game_board, valid_moves, depth, is_max=True)
        
        best_move = ordered_moves[0]
        best_score = -math.inf
        
        for i, move in enumerate(ordered_moves):
            if time.time() > end_time:
                raise TimeoutError()
                
            dir, mtype = move
            next_board = game_board.forecast_move(dir, mtype)
            if not next_board: continue
            
            # Switch perspective for next call
            next_board.reverse_perspective()
            
            score = -self._pvs(next_board, depth - 1, -beta, -alpha, end_time)
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, score)
            if alpha >= beta:
                break
                
        return best_move, best_score

    def _pvs(self, game_board, depth, alpha, beta, end_time):
        """
        Principal Variation Search (NegaScout).
        """
        # 1. Time Check
        if self.nodes_visited % 1000 == 0:
            if time.time() > end_time:
                raise TimeoutError()
        self.nodes_visited += 1
        
        # 2. Transposition Table Lookup
        board_hash = self.tt.compute_hash(game_board)
        tt_entry = self.tt.lookup(board_hash)
        
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == TTFlag.EXACT:
                return tt_entry['score']
            elif tt_entry['flag'] == TTFlag.LOWERBOUND:
                alpha = max(alpha, tt_entry['score'])
            elif tt_entry['flag'] == TTFlag.UPPERBOUND:
                beta = min(beta, tt_entry['score'])
            if alpha >= beta:
                return tt_entry['score']

        # 3. Terminal / Leaf
        if depth <= 0 or game_board.is_game_over():
            # Quiescence? For now just eval
            val = self.evaluator.evaluate(game_board)
            # TT Store
            # Note: Evaluator returns score from perspective of player whose turn it is?
            # Evaluator.evaluate assumes "chicken_player" is us. 
            # In minimax, we swap perspectives. So evaluate always returns "current player advantage".
            return val

        # 4. Move Generation & Ordering
        valid_moves = game_board.get_valid_moves()
        if not valid_moves:
             # No moves -> terminal or pass? Game rules say game over if blocked?
             # Or if blocked, check logic. Board.get_valid_moves handles it.
             # If no moves, usually implies loss or specific state.
             return self.evaluator.evaluate(game_board)
             
        # Use TT move for ordering
        tt_move = tt_entry['best_move'] if tt_entry else None
        ordered_moves = self._order_moves(game_board, valid_moves, depth, is_max=True, tt_move=tt_move)

        best_score = -math.inf
        best_move = None
        
        # 5. Iterate Moves
        alpha_orig = alpha

        for i, move in enumerate(ordered_moves):
            dir, mtype = move
            next_board = game_board.forecast_move(dir, mtype)
            if not next_board: continue
            
            next_board.reverse_perspective()
            
            score = 0
            if i == 0:
                # First move: Full Window
                score = -self._pvs(next_board, depth - 1, -beta, -alpha, end_time)
            else:
                # Late moves: Null Window Search (Check if better than alpha)
                score = -self._pvs(next_board, depth - 1, -alpha - 1, -alpha, end_time)
                if alpha < score < beta:
                    # Research with full window
                    score = -self._pvs(next_board, depth - 1, -beta, -alpha, end_time)
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, score)
            if alpha >= beta:
                break # Cutoff

        # 6. Store in TT
        flag = TTFlag.EXACT
        if best_score <= alpha_orig:
            flag = TTFlag.UPPERBOUND
        elif best_score >= beta:
            flag = TTFlag.LOWERBOUND
        else:
            flag = TTFlag.EXACT
            
        self.tt.store(board_hash, depth, best_score, flag, best_move)
        
        return best_score

    def _order_moves(self, game_board, moves, depth, is_max, tt_move=None):
        """
        Heuristic move ordering.
        """
        scores = []
        for move in moves:
            score = 0
            if move == tt_move:
                score += 10000
            
            mtype = move[1]
            if mtype == enums.MoveType.EGG:
                score += 500
            elif mtype == enums.MoveType.TURD:
                # Corner denial logic in move ordering
                # Check if this turd placement is on a corner
                dir = move[0]
                next_loc = game_board.chicken_player.get_next_loc(dir)
                
                if next_loc is not None:
                    corners = [(0, 0), (0, 7), (7, 0), (7, 7)] # Assuming 8x8 board
                    if next_loc in corners:
                        my_parity = game_board.chicken_player.even_chicken
                        corner_parity = (next_loc[0] + next_loc[1]) % 2
                        if corner_parity != my_parity:
                            score += 5000 # High priority to check this blocking move
            
            # We rely on the evaluation function at the leaves to tell us if a turd is good.
            # At internal nodes, without a flat bonus, turd moves will be searched later (unless TT suggests otherwise),
            # which is fine as they are now considered "expensive/risky" resource usage.
                
            scores.append(score)
            
        # Sort
        zipped = zip(moves, scores)
        ordered = sorted(zipped, key=lambda x: x[1], reverse=True)
        return [x[0] for x in ordered]

