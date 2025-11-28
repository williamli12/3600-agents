from game import board, enums
from typing import Tuple, List, Set
import collections
import math

class Evaluator:
    def __init__(self, trap_belief, visited_locations):
        self.trap_belief = trap_belief
        self.visited_locations = visited_locations # Reference to set in Agent
        self.board_size = 8 # Assumption, will update if dynamic

    def evaluate(self, game_board: board.Board) -> float:
        """
        Evaluate the board state from the current player's perspective.
        Returns a score where higher is better.
        """
        # 1. Terminal States
        if game_board.is_game_over():
            winner = game_board.get_winner()
            if winner == enums.Result.PLAYER:
                return 100000.0
            elif winner == enums.Result.ENEMY:
                return -100000.0
            else:
                return 0.0

        score = 0.0
        
        # 2. Egg Difference (Primary Objective)
        my_eggs = game_board.chicken_player.get_eggs_laid()
        enemy_eggs = game_board.chicken_enemy.get_eggs_laid()
        score += (my_eggs - enemy_eggs) * 1000.0

        # 3. Trapdoor Risk
        my_loc = game_board.chicken_player.get_location()
        risk = self.trap_belief.risk(my_loc)
        if risk == 1.0:
            score -= 100000.0 # Avoid known trapdoors at ALL costs (infinite penalty)
        elif risk > 0.0:
            score -= risk * 5000.0 # High penalty for probable trapdoors

        # 4. Mobility & Blocking
        valid_moves = game_board.get_valid_moves()
        num_moves = len(valid_moves)
        
        if num_moves == 0:
            return -50000.0 # Almost as bad as losing
            
        score += num_moves * 100.0 # Increased from 10.0 to heavily encourage mobility
        
        # Severe penalty for low mobility (danger of being blocked)
        if num_moves == 1:
            score -= 2000.0
        elif num_moves == 2:
            score -= 500.0
        
        # Opponent mobility (we want to restrict them)
        # Use cheap calculation for enemy moves if possible, or standard valid_moves
        enemy_moves = game_board.get_valid_moves(enemy=True)
        enemy_mobility = len(enemy_moves)
        score -= enemy_mobility * 20.0
        
        # Bonus for severe restriction
        if enemy_mobility <= 1:
            score += 1000.0

        # 5. Turd Management
        turds_left = game_board.chicken_player.get_turds_left()
        is_early_game = game_board.turn_count < 30
        
        if is_early_game:
            score += turds_left * 20.0
        else:
            score += turds_left * 5.0
            
        # Check for wasteful edge turds
        for turd_loc in game_board.turds_player:
            if self._is_edge(turd_loc):
                score -= 50.0
        
        # 6. Corner Denial (Prevent opponent from using corners)
        corners = [(0, 0), (0, self.board_size - 1), (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        my_parity = game_board.chicken_player.even_chicken
        
        for corner in corners:
            # Bonus for OWNING the corner (Egg)
            if corner in game_board.eggs_player:
                score += 5000.0 # Ensure owning > approaching
                
            if corner in game_board.turds_player:
                corner_parity = (corner[0] + corner[1]) % 2
                if corner_parity != my_parity:
                    score += 500.0 # Huge bonus for successful denial

        # 7. Exploration (New Areas)
        if my_loc not in self.visited_locations:
            score += 20.0 # Bonus for exploring new squares
            
        # 8. Opponent Proximity (Contextual)
        enemy_loc = game_board.chicken_enemy.get_location()
        dist_to_enemy = abs(my_loc[0] - enemy_loc[0]) + abs(my_loc[1] - enemy_loc[1])
        
        if dist_to_enemy <= 2:
            # Close quarters combat
            if num_moves > enemy_mobility:
                score += 100.0 # Press the advantage
            elif num_moves < enemy_mobility:
                score -= 1000.0 # DANGER: Get out or equalize immediately
            else:
                score += 10.0 # Neutral

        # 9. Reachable Area (Self-Trapping Prevention)
        # OPTIMIZATION: Only check if mobility is low
        if num_moves < 3:
            reachable_count = self._count_reachable(game_board, my_loc, limit=10)
            if reachable_count < 6:
                score -= (6 - reachable_count) * 1000.0 # Heavy penalty for small areas

        # 10. Corner Magnet (Attract to valuable corners)
        # Identify corners that are valid for egg laying and pull Gary towards them
        dist_to_corner = self._dist_to_nearest_valid_corner(game_board)
        if dist_to_corner is not None:
            # Inverse distance bonus. Max bonus for being AT the corner (dist=0).
            # 5000.0 ensures it outweighs many other factors but not traps.
            score += 5000.0 / (dist_to_corner + 1)

        return score

    def _dist_to_nearest_valid_corner(self, game_board: board.Board) -> float:
        """
        Manhattan distance to nearest valid CORNER egg-laying square.
        Optimized for speed (replaced BFS).
        """
        start = game_board.chicken_player.get_location()
        parity = game_board.chicken_player.even_chicken
        
        corners = [(0, 0), (0, self.board_size - 1), (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        min_dist = float('inf')
        found_one = False

        for c in corners:
            # Check parity
            if (c[0] + c[1]) % 2 == parity:
                # Check if blocked/occupied
                # We check if the destination itself is valid for laying.
                if not game_board.is_cell_blocked(c) and c not in game_board.eggs_player and c not in game_board.turds_player:
                     dist = abs(start[0] - c[0]) + abs(start[1] - c[1])
                     if dist < min_dist:
                         min_dist = dist
                         found_one = True
        
        return min_dist if found_one else None

    def _count_reachable(self, game_board: board.Board, start: Tuple[int, int], limit: int) -> int:
        """
        Count reachable squares from start using BFS, up to limit.
        """
        queue = collections.deque([start])
        visited = {start}
        count = 0
        
        while queue and count < limit:
            curr = queue.popleft()
            count += 1
            
            for d in enums.Direction:
                try:
                    next_loc = enums.loc_after_direction(curr, d)
                except ValueError:
                    continue
                    
                if not game_board.is_valid_cell(next_loc):
                    continue
                
                if next_loc in visited:
                    continue
                    
                if game_board.is_cell_blocked(next_loc):
                    continue
                    
                visited.add(next_loc)
                queue.append(next_loc)
                
        return count

    def _is_edge(self, loc: Tuple[int, int]) -> bool:
        x, y = loc
        return x == 0 or x == self.board_size - 1 or y == 0 or y == self.board_size - 1

    def heuristic(self, board):
        return self.evaluate(board)
