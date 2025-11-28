import random
from enum import IntEnum

class TTFlag(IntEnum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

class TranspositionTable:
    def __init__(self, board_size=8):
        self.table = {}
        self.board_size = board_size
        self.zobrist_keys = self._init_zobrist()

    def _init_zobrist(self):
        random.seed(42) # Fixed seed for reproducibility
        keys = {
            'player_pos': [[random.getrandbits(64) for _ in range(self.board_size)] for _ in range(self.board_size)],
            'enemy_pos': [[random.getrandbits(64) for _ in range(self.board_size)] for _ in range(self.board_size)],
            'player_egg': [[random.getrandbits(64) for _ in range(self.board_size)] for _ in range(self.board_size)],
            'enemy_egg': [[random.getrandbits(64) for _ in range(self.board_size)] for _ in range(self.board_size)],
            'player_turd': [[random.getrandbits(64) for _ in range(self.board_size)] for _ in range(self.board_size)],
            'enemy_turd': [[random.getrandbits(64) for _ in range(self.board_size)] for _ in range(self.board_size)],
            'turn_parity': random.getrandbits(64)
        }
        return keys

    def compute_hash(self, board):
        h = 0
        px, py = board.chicken_player.get_location()
        h ^= self.zobrist_keys['player_pos'][px][py]
        ex, ey = board.chicken_enemy.get_location()
        h ^= self.zobrist_keys['enemy_pos'][ex][ey]
        for (x, y) in board.eggs_player:
            h ^= self.zobrist_keys['player_egg'][x][y]
        for (x, y) in board.eggs_enemy:
            h ^= self.zobrist_keys['enemy_egg'][x][y]
        for (x, y) in board.turds_player:
            h ^= self.zobrist_keys['player_turd'][x][y]
        for (x, y) in board.turds_enemy:
            h ^= self.zobrist_keys['enemy_turd'][x][y]
        if board.is_as_turn:
            h ^= self.zobrist_keys['turn_parity']
        return h

    def store(self, board_hash, depth, score, flag, best_move):
        self.table[board_hash] = {
            'depth': depth,
            'score': score,
            'flag': flag,
            'best_move': best_move
        }

    def lookup(self, board_hash):
        return self.table.get(board_hash)

    def clear(self):
        self.table.clear()

