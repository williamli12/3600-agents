from collections.abc import Callable
from typing import List, Set, Tuple
import numpy as np

from game import *
from game.enums import Direction, MoveType, loc_after_direction


class PlayerAgent:
    """
    Ben is a heuristic agent:
    - Tracks trapdoor candidates using sensor data
    - Scores each legal move via a simulated board copy
    - Prefers egg laying, mobility, and avoiding likely trapdoors
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.all_squares = [(x, y) for x in range(self.map_size) for y in range(self.map_size)]

        # For trapdoor A and trapdoor B
        self.trap_candidates = [
            set(self.all_squares),  # candidates for trapdoor 0
            set(self.all_squares),  # candidates for trapdoor 1
        ]

    # ------------- Trapdoor Helpers -------------

    def _hear_zone(self, dx, dy):
        if dx > 2 or dy > 2:
            return False
        if dx == 2 and dy == 2:
            return False
        return True

    def _feel_zone(self, dx, dy):
        return dx <= 1 and dy <= 1

    def _update_trapdoor_candidates(self, board_obj, sensor_data):
        myx, myy = board_obj.chicken_player.get_location()

        for ti in range(2):
            heard, felt = sensor_data[ti]
            old = self.trap_candidates[ti]
            new = set()

            for (x, y) in old:
                dx, dy = abs(x - myx), abs(y - myy)

                in_hear = self._hear_zone(dx, dy)
                in_feel = self._feel_zone(dx, dy)

                # Deterministic pruning approx
                if heard and not in_hear: 
                    continue
                if not heard and in_hear:
                    continue
                if felt and not in_feel:
                    continue
                if not felt and in_feel:
                    continue

                new.add((x, y))

            # only shrink if not empty
            if new:
                self.trap_candidates[ti] = new

    def _trap_risk(self, loc):
        risk = 0.0
        for cand in self.trap_candidates:
            if loc in cand:
                risk += 1 / max(1, len(cand))
        return risk

    # ------------- Evaluation Helpers -------------

    def _egg_potential(self, b):
        x0, y0 = b.chicken_player.get_location()
        score = 0.0

        try:
            if b.can_lay_egg_at_loc((x0, y0)):
                score += 1
        except:
            pass

        for d in Direction:
            nx, ny = loc_after_direction((x0, y0), d)
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                try:
                    if b.can_lay_egg_at_loc((nx, ny)):
                        score += 0.4
                except:
                    pass

        return score

    def _evaluate(self, b):
        my_eggs = len(b.eggs_player)
        them_eggs = len(b.eggs_enemy)
        egg_diff = my_eggs - them_eggs

        mobility = len(b.get_valid_moves(enemy=False))
        egg_pot = self._egg_potential(b)

        loc = b.chicken_player.get_location()
        trap_penalty = 8 * self._trap_risk(loc)

        return (10 * egg_diff) + (0.7 * mobility) + (2 * egg_pot) - trap_penalty

    # ------------- Main Move Function -------------

    def play(self, board_obj, sensor_data, time_left):
        # Update trapdoor logic
        self._update_trapdoor_candidates(board_obj, sensor_data)

        moves = board_obj.get_valid_moves()
        if not moves:
            return Direction.UP, MoveType.PLAIN

        best_move = None
        best_score = -1e18

        for (direction, move_type) in moves:
            b_copy = board_obj.get_copy()
            ok = b_copy.apply_move(direction, move_type, timer=0.0, check_ok=True)
            if not ok:
                continue

            score = self._evaluate(b_copy)
            score += 0.001 * np.random.random()  # tie breaker

            if score > best_score:
                best_score = score
                best_move = (direction, move_type)

        if best_move is None:
            best_move = moves[np.random.randint(len(moves))]

        return best_move
