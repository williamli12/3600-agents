# MaxBeater - Advanced Chicken Game Agent

A high-performance AI agent for the ByteFight Chicken game, combining Monte Carlo Tree Search (MCTS) with learned value estimation and sophisticated heuristics.

## 🏗️ Architecture Overview

MaxBeater uses a **layered architecture** with clean separation between runtime and training components:

```
Runtime Stack (NumPy only):
┌─────────────────────────────────────────┐
│  agent.py (Orchestrator)                │
│  ├─ TrapdoorBelief (Bayesian tracking)  │
│  ├─ Evaluator (heuristic + value net)   │
│  ├─ MCTSSearch (main search engine)     │
│  └─ FallbackSearch (fast search)        │
└─────────────────────────────────────────┘

Training Pipeline (PyTorch):
┌─────────────────────────────────────────┐
│  train_value_model.py                   │
│  → Self-play games                      │
│  → Neural network training              │
│  → Export to value_weights.npz          │
└─────────────────────────────────────────┘
```

---

## 📂 File Structure

| File | Purpose | Runtime Required |
|------|---------|------------------|
| `agent.py` | Main orchestrator, entry point | ✅ Yes |
| `belief.py` | Bayesian trapdoor belief tracking | ✅ Yes |
| `features.py` | Feature extraction (heuristic + NN input) | ✅ Yes |
| `evaluator.py` | Combined evaluation (heuristic + value model) | ✅ Yes |
| `value_model_runtime.py` | NumPy-only neural network inference | ✅ Yes |
| `search_mcts.py` | MCTS with UCT selection | ✅ Yes |
| `search_fallback.py` | Fast greedy/two-ply search | ✅ Yes |
| `train_value_model.py` | PyTorch training script (offline) | ❌ No |
| `value_weights.npz` | Trained neural network weights | ⚠️ Optional |
| `__init__.py` | Package exports | ✅ Yes |

---

## 🧠 Component Details

### 1. TrapdoorBelief (`belief.py`)

**Purpose**: Track probability distributions over trapdoor locations using Bayesian inference.

**Key Features**:
- Maintains separate beliefs for even/odd parity trapdoors
- Initializes with game-rule-based priors (center-weighted)
- Updates beliefs using sensor data (heard/felt) via Bayes' rule
- Provides risk assessment for positions

**Key Methods**:
```python
belief.update(board, sensor_data)  # Update beliefs
belief.risk(coord)                 # Get trapdoor probability at coord
belief.max_risk_in_radius(coord)   # Max risk in neighborhood
```

---

### 2. Feature Extraction (`features.py`)

**Purpose**: Convert board states into feature vectors for evaluation.

**Two Feature Types**:

1. **Scalar Features (26 dims)**: Hand-crafted features
   - Egg differential, mobility, turns left
   - Positional features (center distance, corner proximity)
   - Trapdoor risk metrics
   - Turd resource counts

2. **Board Tensor (14×8×8)**: Spatial representation for neural network
   - Channels: chicken positions, eggs, turds, trapdoor beliefs, parity maps, corners, etc.

**Key Functions**:
```python
extract_scalar_features(board, trap_belief)  # Returns (26,) array
encode_board_tensor(board, trap_belief)      # Returns (14, 8, 8) array
```

---

### 3. Value Model Runtime (`value_model_runtime.py`)

**Purpose**: NumPy-only neural network for value estimation.

**Architecture**:
```
Input (1050)  =  14×8×8 (board tensor) + 26 (scalars)
    ↓
Hidden1 (256) + ReLU
    ↓
Hidden2 (128) + ReLU
    ↓
Output (1) + tanh  →  [-1, 1]
```

**Features**:
- Pure NumPy implementation (no PyTorch dependency)
- Loads weights from `value_weights.npz`
- Returns 0.0 if weights not available (heuristic-only mode)

**Usage**:
```python
model = ValueModelRuntime()
value = model.forward(features)  # Returns scalar in [-1, 1]
```

---

### 4. Evaluator (`evaluator.py`)

**Purpose**: Combine hand-crafted heuristics with learned value function.

**Heuristic Components**:
| Component | Weight | Description |
|-----------|--------|-------------|
| Egg differential | 1000 | Primary objective (×1.5 in endgame) |
| Corner bonus | 200 | Corners give 3× eggs |
| Blocking | ±500 | Blocking enemy / being blocked |
| Mobility | 15 | # of valid moves |
| Trapdoor risk | 150 | Risk at current position |
| Turd resource | 30 | Turd count differential |
| Center control | 10 | Strategic positioning (midgame) |

**Blending Strategy**:
- Extreme heuristic scores (>2000): Trust heuristic 100%
- Moderate scores (1000-2000): 70% heuristic, 30% value model
- Uncertain positions (<1000): 50-50 blend

**Key Methods**:
```python
evaluator.heuristic(board)       # Hand-crafted evaluation
evaluator.evaluate(board)        # Combined heuristic + value model
evaluator.quick_evaluate(board)  # Fast eval for move ordering
```

---

### 5. MCTS Search (`search_mcts.py`)

**Purpose**: Main search engine using Monte Carlo Tree Search with UCT.

**Algorithm**:
1. **Selection**: Traverse tree using UCT formula
   ```
   UCB(node) = Q + c_puct × P × √(N_parent) / (1 + N)
   ```
2. **Expansion**: Create children for unvisited nodes
3. **Evaluation**: Use value network (no random rollouts)
4. **Backpropagation**: Update statistics, alternating sign for minimax

**Parameters**:
- `c_puct = 1.5`: Exploration constant
- `max_simulations = 2000`: Maximum simulations per move
- `time_safety_margin = 3.0s`: Reserved time buffer

**Perspective Handling**:
```python
# CRITICAL: After applying move, reverse perspective
new_board = board.forecast_move(direction, move_type)
new_board.reverse_perspective()  # Child = opponent's perspective
```

---

### 6. Fallback Search (`search_fallback.py`)

**Purpose**: Fast search for time-critical situations.

**Modes**:
- **One-ply greedy** (< 2s remaining): Immediate evaluation only
- **Two-ply** (≥ 2s remaining): Our move + enemy's best response

**Usage**: Automatically triggered when time < 15s

---

### 7. Agent Orchestrator (`agent.py`)

**Purpose**: Main entry point that coordinates all components.

**Decision Logic**:
```python
if remaining_time < 5s:
    use FallbackSearch (greedy)
elif remaining_time < 15s:
    use FallbackSearch (two-ply)
else:
    use MCTSSearch (full search)
```

**Interface** (required by game engine):
```python
class PlayerAgent:
    def __init__(self, board, time_left):
        # Initialize all components
    
    def play(self, board, sensor_data, time_left):
        # Update beliefs, choose move, return (Direction, MoveType)
```

---

## 🏋️ Training Pipeline

### Training Script (`train_value_model.py`)

**Purpose**: Generate `value_weights.npz` using PyTorch (offline only).

**Process**:
1. **Self-play**: Generate games using random or greedy play
2. **Data collection**: Extract (state_features, outcome_value) pairs
3. **Training**: Train MLP using MSE loss
4. **Export**: Convert PyTorch weights to NumPy format

**Usage** (on PACE cluster):
```bash
cd 3600-agents/MaxBeater

# Basic training (500 games, 50 epochs)
python train_value_model.py

# Advanced training
python train_value_model.py \
    --games 2000 \
    --epochs 100 \
    --batch-size 512 \
    --lr 0.0005 \
    --use-mcts \
    --device cuda
```

**Output**: `value_weights.npz` with keys `W1, b1, W2, b2, W3, b3`

**Note**: Training is optional. Agent works with heuristic-only if weights not available.

---

## 🚀 Usage

### Runtime Deployment

The agent is automatically instantiated by the game engine:

```python
# Game engine calls:
agent = PlayerAgent(initial_board, time_left_function)

# Each turn:
move = agent.play(current_board, sensor_data, time_left_function)
```

**No manual setup required!** The agent is ready to compete as-is.

---

### Testing Locally

```bash
cd 3600-agents/MaxBeater
python agent.py  # Runs built-in test
```

Or use the game engine's runner:
```bash
cd engine
python run_local_agents.py --agent1 MaxBeater --agent2 MinimaxAgent
```

---

## 🔧 Configuration & Tuning

### MCTS Parameters (`search_mcts.py`)

```python
self.c_puct = 1.5              # Exploration constant (higher = more exploration)
self.max_simulations = 2000    # Max simulations per move
self.time_safety_margin = 3.0  # Time reserve (seconds)
```

### Heuristic Weights (`evaluator.py`)

```python
self.weights = {
    'egg_diff': 1000.0,
    'corner_egg_bonus': 200.0,
    'blocking_bonus': 500.0,
    # ... etc
}
```

### Time Thresholds (`agent.py`)

```python
self.low_time_threshold = 15.0     # Switch to fallback
self.critical_time_threshold = 5.0  # Emergency greedy mode
```

---

## 🎯 Strategy Highlights

### Early Game (Turns 1-15)
- **Explore center**: Control strategic positions
- **Conservative turd usage**: Save turds for critical moments
- **Egg maximization**: Focus on regular eggs, set up for corners

### Mid Game (Turns 16-30)
- **Corner exploitation**: Prioritize corner positions (3× eggs)
- **Blocking attempts**: Use turds to limit enemy mobility
- **Trapdoor avoidance**: Use belief tracker to avoid high-risk areas

### End Game (Turns 31-40)
- **Aggressive egg laying**: Maximize egg count
- **Time management**: Balance search depth with time constraints
- **Defensive play**: Prevent enemy from catching up

---

## 📊 Expected Performance

### Strengths
- ✅ Strong heuristic evaluation (beats basic minimax)
- ✅ Adaptive search (MCTS for deep analysis, fallback for speed)
- ✅ Probabilistic trapdoor reasoning
- ✅ Time-aware decision making
- ✅ Optional learned value function enhancement

### Weaknesses
- ⚠️ MCTS slower than alpha-beta for shallow searches
- ⚠️ Value model requires training data (works without it)
- ⚠️ Limited domain knowledge vs. hand-tuned specialists

### vs. Baseline Agents
| Opponent | Expected Win Rate |
|----------|------------------|
| Random | 95%+ |
| MinimaxAgent (basic) | 70-80% |
| Bob (alpha-beta + heuristics) | 50-60% |
| AlphaChicken (with training) | 40-50% |

---

## 🐛 Troubleshooting

### "Weights file not found"
- **Expected behavior**: Agent runs with heuristic-only mode
- **Solution**: Train model and copy `value_weights.npz` to MaxBeater directory

### "Agent timing out"
- **Check**: Time thresholds in `agent.py`
- **Solution**: Reduce `max_simulations` or increase `time_safety_margin`

### "Invalid move returned"
- **Cause**: Bug in perspective handling
- **Fallback**: Agent automatically selects random valid move

### Import errors
- **Cause**: Game module not in path
- **Solution**: Ensure running from correct directory with game engine accessible

---

## 🔬 Future Enhancements

1. **Policy Network**: Add policy head to guide MCTS priors
2. **Opening Book**: Pre-computed strong openings
3. **Endgame Solver**: Exact solutions for last few turns
4. **Curriculum Training**: Train against progressively stronger opponents
5. **Data Augmentation**: Board symmetries for more training data
6. **Hyperparameter Tuning**: Grid search for optimal weights

---

## 📚 References

- **MCTS**: Browne et al. (2012) - "A Survey of Monte Carlo Tree Search Methods"
- **UCT**: Kocsis & Szepesvári (2006) - "Bandit based Monte-Carlo Planning"
- **AlphaZero**: Silver et al. (2017) - "Mastering Chess and Shogi by Self-Play"

---

## 👥 Development

**CS 3600 - Artificial Intelligence**  
**Agent**: MaxBeater  
**Strategy**: MCTS + Learned Evaluation + Heuristics  
**Status**: Production Ready 🚀

---

## 📝 License & Usage

This code is for educational purposes as part of CS 3600.
Free to use and modify for the course project.

---

**Good luck in the competition! 🐔🥚**


