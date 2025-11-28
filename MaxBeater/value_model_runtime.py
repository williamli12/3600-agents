"""
value_model_runtime.py - NumPy-only neural network for value estimation

This module provides a lightweight MLP that runs entirely in NumPy.
Weights are loaded from value_weights.npz (created by train_value_model.py).

Network architecture:
  Input: Flattened board tensor (14*8*8) + scalar features (26) = 1050 dims
  Hidden1: 256 units, ReLU
  Hidden2: 128 units, ReLU
  Output: 1 unit, tanh -> [-1, 1]

Value interpretation:
  +1.0 = very likely to win (large positive egg differential)
  -1.0 = very likely to lose (large negative egg differential)
   0.0 = even position
"""

from __future__ import annotations
from typing import Optional
import os
import numpy as np


class ValueModelRuntime:
    """
    Simple NumPy-only MLP for state value estimation.
    
    Loads weights from value_weights.npz if present.
    If weights are not found, returns 0.0 for all states (neutral evaluation).
    """
    
    def __init__(self, weights_path: Optional[str] = None):
        """
        Initialize the value model.
        
        Args:
            weights_path: Optional path to weights file. If None, looks for
                         value_weights.npz in the same directory as this file.
        """
        self.weights_loaded = False
        
        # Weights and biases for 3-layer MLP
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.W3: Optional[np.ndarray] = None
        self.b3: Optional[np.ndarray] = None
        
        # Try to load weights
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), "value_weights.npz")
        
        self._maybe_load_weights(weights_path)
    
    def _maybe_load_weights(self, path: str) -> None:
        """
        Attempt to load weights from .npz file.
        
        Expected keys: W1, b1, W2, b2, W3, b3
        """
        if not os.path.exists(path):
            print(f"[ValueModel] Weights file not found: {path}")
            print("[ValueModel] Using zero-value fallback (heuristic only)")
            return
        
        try:
            data = np.load(path)
            
            # Load all weights
            self.W1 = data["W1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.W2 = data["W2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)
            self.W3 = data["W3"].astype(np.float32)
            self.b3 = data["b3"].astype(np.float32)
            
            self.weights_loaded = True
            print(f"[ValueModel] Loaded weights from {path}")
            print(f"[ValueModel] Architecture: {self.W1.shape[1]} -> {self.W1.shape[0]} -> {self.W2.shape[0]} -> 1")
            
        except Exception as e:
            print(f"[ValueModel] Error loading weights: {e}")
            print("[ValueModel] Using zero-value fallback (heuristic only)")
            self.weights_loaded = False
    
    def forward(self, x: np.ndarray) -> float:
        """
        Run forward pass through the network.
        
        Args:
            x: Input feature vector (1D array)
               Expected shape: (1050,) = 14*8*8 + 26
        
        Returns:
            Scalar value in [-1, 1]
        """
        if not self.weights_loaded:
            return 0.0
        
        try:
            # Ensure input is float32 and 1D
            x = x.astype(np.float32).flatten()
            
            # Layer 1: Linear + ReLU
            h1 = np.dot(self.W1, x) + self.b1
            h1 = np.maximum(0, h1)  # ReLU
            
            # Layer 2: Linear + ReLU
            h2 = np.dot(self.W2, h1) + self.b2
            h2 = np.maximum(0, h2)  # ReLU
            
            # Output layer: Linear + tanh
            out = np.dot(self.W3, h2) + self.b3
            
            # Squash to [-1, 1] using tanh
            value = float(np.tanh(out[0]))
            
            return value
            
        except Exception as e:
            print(f"[ValueModel] Forward pass error: {e}")
            return 0.0
    
    def is_loaded(self) -> bool:
        """Check if weights were successfully loaded."""
        return self.weights_loaded


# Utility function for creating a dummy weights file for testing
def create_dummy_weights(output_path: str, input_dim: int = 1050) -> None:
    """
    Create a dummy weights file with random initialization.
    Useful for testing before training is complete.
    
    Args:
        output_path: Path to save the weights file
        input_dim: Input dimension (14*8*8 + 26 = 1050)
    """
    # Simple 3-layer architecture
    hidden1_size = 256
    hidden2_size = 128
    output_size = 1
    
    # Xavier initialization
    W1 = np.random.randn(hidden1_size, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden1_size, dtype=np.float32)
    
    W2 = np.random.randn(hidden2_size, hidden1_size).astype(np.float32) * np.sqrt(2.0 / hidden1_size)
    b2 = np.zeros(hidden2_size, dtype=np.float32)
    
    W3 = np.random.randn(output_size, hidden2_size).astype(np.float32) * np.sqrt(2.0 / hidden2_size)
    b3 = np.zeros(output_size, dtype=np.float32)
    
    # Save
    np.savez(output_path, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    print(f"[ValueModel] Created dummy weights at {output_path}")


