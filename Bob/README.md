# Bob - Advanced CS3600 Chicken Game AI Agent

Bob is a sophisticated AI agent for the CS3600 ByteFight chicken game, implementing classical AI techniques including adversarial search, probabilistic reasoning, and optional learned value functions.

## Architecture Overview

The agent is structured into modular components, each handling a specific aspect of decision-making:

```
agent.py                 # Thin orchestration layer (PlayerAgent class)
├── trapdoor_belief.py   # Bayesian belief tracking for trapdoors
├── evaluation.py        # Board evaluation (heuristic + optional neural net)
├── search.py            # Alpha-beta search with iterative deepening
└── features.py          # Feature extraction from game state
```

### Components

#### 1. **agent.py** - Main Entry Point
- **Purpose**: Thin orchestrator that delegates to specialized modules
- **Key Method**: `play(board, sensor_data, time_left)` - called by game engine each turn
- **Responsibilities**:
  - Update trapdoor beliefs with sensor data
  - Invoke search engine to select best move
  - Validate moves and provide fallback

#### 2. **trapdoor_belief.py** - Probabilistic Reasoning
- **Purpose**: Maintain Bayesian belief distributions over trapdoor locations
- **Key Features**:
  - Two probability grids (one per trapdoor parity)
  - Ring-based priors (edges: low weight, center: high weight)
  - Bayesian updates using sensor observations (heard/felt)
  - Risk assessment for positions
- **Methods**:
  - `update(board, sensor_data)` - Update beliefs with new observations
  - `risk(coord)` - Get combined trapdoor probability at position
  - `max_risk_in_radius(center, radius)` - Find max risk in area

#### 3. **features.py** - State Representation
- **Purpose**: Extract numeric features from board state for evaluation
- **Feature Categories** (29 features total):
  - **Eggs**: Counts and differences
  - **Turns**: Remaining turns and tempo
  - **Mobility**: Valid moves for both players
  - **Trapdoor Risk**: Current position and nearby
  - **Spatial Control**: Distance to center, corners
  - **Turds**: Remaining and placed
  - **Time**: Remaining time for both players
  - **Game State**: Terminal state indicators

#### 4. **evaluation.py** - Position Evaluation
- **Purpose**: Score board positions from current player's perspective
- **Key Features**:
  - Hand-crafted heuristic combining multiple strategic factors
  - Optional PyTorch value network (if `value_net.pt` exists)
  - Hybrid evaluation: blend heuristic + neural net predictions
- **Heuristic Factors**:
  - **Egg Difference**: Primary objective (weight: 100)
  - **Mobility**: Avoid getting blocked (weight: 5 per move)
  - **Trapdoor Risk**: Penalize high-risk squares (weight: -800 at position)
  - **Corner Control**: Reward corner positions for 3x eggs
  - **Terminal States**: Detect wins/losses (±10,000)
  - **Blocked State**: Heavily penalize no valid moves (-5,000)

#### 5. **search.py** - Move Selection
- **Purpose**: Alpha-beta search with iterative deepening
- **Key Features**:
  - Time-bounded search with safety margins
  - Iterative deepening (depth 1 → 6+)
  - Alpha-beta pruning for efficiency
  - Move ordering for better pruning
  - Greedy fallback when time is critical
- **Search Strategy**:
  - Per-move time budget based on remaining turns
  - Always returns valid moves
  - Statistics tracking (nodes searched, max depth)

#### 6. **train_value_net.py** - Optional Training (Scaffold)
- **Purpose**: Offline training script for value network
- **Status**: Scaffold only - not required for agent to function
- **Workflow** (when implemented):
  1. Generate self-play training data
  2. Extract features from game positions
  3. Train neural network to predict outcomes
  4. Save as `value_net.pt`

## Safety Features

1. **No Invalid Moves**: Always selects from `board.get_valid_moves()`
2. **Time Management**: 
   - Safety margin (2 seconds) to prevent timeouts
   - Per-move budgets proportional to remaining turns
   - Early termination when time is low
3. **Robust Fallbacks**: 
   - Greedy evaluation if search time expires
   - Random valid move if all else fails
4. **Numerical Stability**: 
   - Epsilon values prevent exact zeros in probabilities
   - Normalized belief distributions

## Strategic Priorities

1. **Maximize Egg Advantage**: Primary objective
2. **Avoid Getting Blocked**: Critical (-5 eggs penalty)
3. **Avoid Trapdoors**: Especially high-probability squares (-4 eggs penalty)
4. **Control Corners**: Worth 3x eggs
5. **Maintain Mobility**: Keep move options open
6. **Strategic Turd Placement**: Block opponent while preserving our mobility

## Configuration

Key parameters in `search.py`:
- `max_search_depth = 6`: Maximum search depth
- `time_safety_margin = 2.0`: Seconds to reserve for safety
- Can be tuned based on performance analysis

## Dependencies

- **Required**: `numpy` (for numerical operations)
- **Optional**: `torch` (for value network, gracefully degrades if unavailable)
- **Game Engine**: `game` module (provided by assignment)

## Usage

The agent is automatically loaded by the game engine. No manual invocation needed.

```python
from Bob import PlayerAgent

# Engine will instantiate and call:
agent = PlayerAgent(board, time_left)
move = agent.play(board, sensor_data, time_left)
```

## Performance Characteristics

- **Search Depth**: Typically 3-6 ply depending on position complexity
- **Branching Factor**: ~8-12 moves per position
- **Nodes/Second**: 100-1000+ depending on evaluation complexity
- **Time Per Move**: ~0.1-0.5 seconds (varies with remaining turns)

## Future Enhancements

1. **Transposition Tables**: Cache evaluated positions
2. **Opening Book**: Pre-computed early-game moves
3. **Endgame Database**: Optimal play in simplified positions
4. **Full Value Network Training**: Complete self-play training pipeline
5. **Quiescence Search**: Extend search at tactical positions
6. **Null-Move Pruning**: Additional search optimizations

## Author Notes

This agent balances classical AI techniques with modern machine learning approaches:
- **Classical**: Alpha-beta search, Bayesian reasoning, heuristic evaluation
- **Modern**: Optional neural network value function (when trained)
- **Hybrid**: Combines strengths of both approaches

The modular design allows easy testing and improvement of individual components.

