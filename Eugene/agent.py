from collections.abc import Callable
from typing import List, Tuple
import numpy as np

from game import *
from game.enums import Direction, MoveType


class PlayerAgent:
    """
    Eugene uses a simple utility based strategy with one step lookahead.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE

    def _evaluate_simple(self, b: board.Board) -> float:
        """
        Basic utility function:
        - egg difference
        - mobility
        """
        my_eggs = len(b.eggs_player)
        their_eggs = len(b.eggs_enemy)
        egg_diff = my_eggs - their_eggs

        my_moves = len(b.get_valid_moves(enemy=False))

        value = 10.0 * egg_diff + 0.5 * my_moves
        return value

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        moves = board.get_valid_moves()
        if not moves:
            return Direction.UP, MoveType.PLAIN

        best_move = None
        best_score = float("-1e18")

        for direction, move_type in moves:
            b_copy = board.get_copy()
            ok = b_copy.apply_move(direction, move_type, timer=0.0, check_ok=True)
            if not ok:
                continue

            score = self._evaluate_simple(b_copy)

            # Slight bias toward egg moves
            if move_type == MoveType.EGG:
                score += 1.0

            # Tiny noise to break ties
            score += 0.001 * np.random.random()

            if score > best_score:
                best_score = score
                best_move = (direction, move_type)

        if best_move is None:
            best_move = moves[np.random.randint(len(moves))]

        return best_move
