# transposition.py
"""
Transposition table using Zobrist hashing for efficient position caching.
Stores previously evaluated positions to avoid redundant computation.
"""

from typing import Optional, Tuple
import numpy as np
from game import board, enums


class ZobristHasher:
    """
    Zobrist hashing for board positions.
    Generates unique hash keys for game states using XOR operations.
    """
    
    def __init__(self, board_size: int = 8, seed: int = 42):
        """
        Initialize Zobrist hash tables.
        
        Args:
            board_size: Size of the game board
            seed: Random seed for reproducible hash values
        """
        self.board_size = board_size
        rng = np.random.default_rng(seed)
        
        # Generate random 64-bit integers for each possible board state
        # We need hashes for:
        # - Player chicken at each position
        # - Enemy chicken at each position  
        # - Player egg at each position
        # - Enemy egg at each position
        # - Player turd at each position
        # - Enemy turd at each position
        # - Turn counter (bucketed)
        # - Whose turn it is
        
        n_positions = board_size * board_size
        
        self.player_pos = rng.integers(0, 2**63, size=n_positions, dtype=np.int64)
        self.enemy_pos = rng.integers(0, 2**63, size=n_positions, dtype=np.int64)
        self.player_eggs = rng.integers(0, 2**63, size=n_positions, dtype=np.int64)
        self.enemy_eggs = rng.integers(0, 2**63, size=n_positions, dtype=np.int64)
        self.player_turds = rng.integers(0, 2**63, size=n_positions, dtype=np.int64)
        self.enemy_turds = rng.integers(0, 2**63, size=n_positions, dtype=np.int64)
        self.turn_bucket = rng.integers(0, 2**63, size=10, dtype=np.int64)  # Bucket turns into 10 groups
        self.side_to_move = rng.integers(0, 2**63, dtype=np.int64)
    
    def _pos_to_index(self, pos: Tuple[int, int]) -> int:
        """Convert (x, y) position to flat index."""
        return pos[0] * self.board_size + pos[1]
    
    def hash(self, game_board: board.Board) -> int:
        """
        Compute Zobrist hash for the current board state.
        
        Args:
            game_board: Board state to hash
            
        Returns:
            64-bit hash value
        """
        h = np.int64(0)
        
        # Player chicken position
        player_loc = game_board.chicken_player.get_location()
        h ^= self.player_pos[self._pos_to_index(player_loc)]
        
        # Enemy chicken position
        enemy_loc = game_board.chicken_enemy.get_location()
        h ^= self.enemy_pos[self._pos_to_index(enemy_loc)]
        
        # Player eggs
        for egg_pos in game_board.eggs_player:
            h ^= self.player_eggs[self._pos_to_index(egg_pos)]
        
        # Enemy eggs
        for egg_pos in game_board.eggs_enemy:
            h ^= self.enemy_eggs[self._pos_to_index(egg_pos)]
        
        # Player turds
        for turd_pos in game_board.turds_player:
            h ^= self.player_turds[self._pos_to_index(turd_pos)]
        
        # Enemy turds
        for turd_pos in game_board.turds_enemy:
            h ^= self.enemy_turds[self._pos_to_index(turd_pos)]
        
        # Turn counter (bucketed to reduce hash space)
        turn_idx = min(game_board.turn_count // 8, 9)  # 10 buckets (0-9)
        h ^= self.turn_bucket[turn_idx]
        
        # Side to move (perspective matters)
        if not game_board.is_as_turn:
            h ^= self.side_to_move
        
        return int(h)


class TTEntry:
    """
    Transposition table entry storing evaluation data for a position.
    """
    
    # Entry types
    EXACT = 0       # Exact value
    LOWER_BOUND = 1 # Alpha cutoff (value >= stored value)
    UPPER_BOUND = 2 # Beta cutoff (value <= stored value)
    
    __slots__ = ['hash_key', 'depth', 'value', 'flag', 'best_move', 'age']
    
    def __init__(
        self, 
        hash_key: int, 
        depth: int, 
        value: float, 
        flag: int,
        best_move: Optional[Tuple[enums.Direction, enums.MoveType]] = None,
        age: int = 0
    ):
        """
        Create a transposition table entry.
        
        Args:
            hash_key: Zobrist hash of the position
            depth: Search depth at which this was evaluated
            value: Evaluation score
            flag: Entry type (EXACT, LOWER_BOUND, or UPPER_BOUND)
            best_move: Best move found at this position
            age: Search age/iteration for replacement policy
        """
        self.hash_key = hash_key
        self.depth = depth
        self.value = value
        self.flag = flag
        self.best_move = best_move
        self.age = age


class TranspositionTable:
    """
    Transposition table for caching evaluated positions.
    Uses Zobrist hashing with depth-preferred replacement.
    """
    
    def __init__(self, max_size_mb: int = 256, board_size: int = 8):
        """
        Initialize the transposition table.
        
        Args:
            max_size_mb: Maximum memory usage in megabytes
            board_size: Size of the game board
        """
        # Estimate entry size (rough approximation)
        bytes_per_entry = 64  # Conservative estimate
        max_entries = (max_size_mb * 1024 * 1024) // bytes_per_entry
        
        # Use power of 2 for efficient modulo
        self.size = 2 ** int(np.log2(max_entries))
        
        self.table = [None] * self.size
        self.hasher = ZobristHasher(board_size=board_size)
        self.current_age = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.collisions = 0
    
    def clear(self):
        """Clear the transposition table."""
        self.table = [None] * self.size
        self.current_age += 1
        self.hits = 0
        self.misses = 0
        self.collisions = 0
    
    def increment_age(self):
        """Increment search age (call at start of each move search)."""
        self.current_age += 1
    
    def get_hash(self, game_board: board.Board) -> int:
        """Get Zobrist hash for a board position."""
        return self.hasher.hash(game_board)
    
    def store(
        self,
        game_board: board.Board,
        depth: int,
        value: float,
        flag: int,
        best_move: Optional[Tuple[enums.Direction, enums.MoveType]] = None
    ):
        """
        Store a position evaluation in the transposition table.
        
        Args:
            game_board: Board position
            depth: Search depth
            value: Evaluation score
            flag: Entry type
            best_move: Best move found
        """
        hash_key = self.hasher.hash(game_board)
        index = hash_key % self.size
        
        existing = self.table[index]
        
        # Replacement policy: replace if deeper search or same position or old age
        should_replace = (
            existing is None or
            existing.hash_key == hash_key or  # Same position, update
            depth >= existing.depth or        # Deeper search
            (self.current_age - existing.age) > 2  # Old entry
        )
        
        if should_replace:
            if existing is not None and existing.hash_key != hash_key:
                self.collisions += 1
            
            self.table[index] = TTEntry(
                hash_key=hash_key,
                depth=depth,
                value=value,
                flag=flag,
                best_move=best_move,
                age=self.current_age
            )
    
    def probe(
        self,
        game_board: board.Board
    ) -> Optional[TTEntry]:
        """
        Look up a position in the transposition table.
        
        Args:
            game_board: Board position
            
        Returns:
            TTEntry if found, None otherwise
        """
        hash_key = self.hasher.hash(game_board)
        index = hash_key % self.size
        
        entry = self.table[index]
        
        if entry is not None and entry.hash_key == hash_key:
            self.hits += 1
            return entry
        else:
            self.misses += 1
            return None
    
    def get_stats(self) -> dict:
        """Return statistics about table performance."""
        total_probes = self.hits + self.misses
        hit_rate = self.hits / total_probes if total_probes > 0 else 0.0
        
        filled = sum(1 for entry in self.table if entry is not None)
        fill_rate = filled / self.size
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'collisions': self.collisions,
            'fill_rate': fill_rate,
            'size': self.size
        }

