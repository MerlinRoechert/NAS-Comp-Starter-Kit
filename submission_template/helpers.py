"""
helpers.py - Cell-based search space, training-free proxy scores (NASWOT, SynFlow),
and utility functions for the NAS pipeline.
"""

import math
import copy
import random
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TIME UTILITIES
def div_remainder(n, interval):
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder

def show_time(seconds):
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)
        
# CELL OPERATIONS
class Identity(nn.Module):
    """Skip connection (identity mapping)."""
    def forward(self, x):
        return x

class ZeroOp(nn.Module):
    """Zero operation (returns zeros)."""
    def forward(self, x):
        return torch.zeros_like(x)

class ConvBnRelu(nn.Module):
    """Conv to BatchNorm to ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size, 
                      stride=stride, 
                      padding=padding, 
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)

class AvgPoolOp(nn.Module):
    """Average pooling with same spatial dims."""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.op = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        return self.op(x)

class MaxPoolOp(nn.Module):
    """Max pooling with same spatial dims."""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.op = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        return self.op(x)

# Operation factory: maps operation name to constructor
# Each constructor takes (in_channels, out_channels) and returns a module
OPERATIONS = {
    'conv3x3': lambda c_in, c_out: ConvBnRelu(c_in, c_out, 3),
    'conv5x5': lambda c_in, c_out: ConvBnRelu(c_in, c_out, 5),
    'sep3x3': lambda c_in, c_out: SepConv(c_in, c_out, 3),
    'sep5x5': lambda c_in, c_out: SepConv(c_in, c_out, 5),
    'avg_pool': lambda c_in, c_out: AvgPoolOp(3),
    'max_pool': lambda c_in, c_out: MaxPoolOp(3),
    'skip': lambda c_in, c_out: Identity(),
}

# Operation names available for search
OP_NAMES = list(OPERATIONS.keys())

# CELL AND NETWORK ARCHITECTURE
class Cell(nn.Module):
    """
    A cell with N nodes. Each node applies one operation to one of the previous
    node outputs and sums them. This is a simplified cell structure for fast search.
    """
    def __init__(self, n_nodes, in_channels, out_channels, ops_per_node):
        """
        Args:
            n_nodes: number of intermediate nodes in the cell
            in_channels: input channels
            out_channels: output channels
            ops_per_node: list of (op_name, input_node_idx) for each node
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Preprocess input to match out_channels
        self.preprocess = ConvBnRelu(in_channels, out_channels, 1)

        # Create operations for each node
        self.ops = nn.ModuleList()
        for op_name, _ in ops_per_node:
            op = OPERATIONS[op_name](out_channels, out_channels)
            self.ops.append(op)

        self.ops_per_node = ops_per_node

    def forward(self, x):
        # Preprocess input
        s0 = self.preprocess(x)
        states = [s0]

        for i, (op_name, input_idx) in enumerate(self.ops_per_node):
            inp = states[input_idx]
            out = self.ops[i](inp)
            states.append(out)

        # Output: mean of all intermediate node outputs (exclude input)
        return torch.mean(torch.stack(states[1:], dim=0), dim=0)

class CellNetwork(nn.Module):
    """
    A network built from stacked cells with optional downsampling between stages.
    """
    def __init__(self, in_channels, num_classes, cell_config, n_cells=3,
                 init_channels=32, channel_multiplier=2):
        """
        Args:
            in_channels: number of input image channels
            num_classes: number of output classes
            cell_config: list of (op_name, input_node_idx) for cell construction
            n_cells: total number of cells
            init_channels: initial channel width
            channel_multiplier: multiply channels at each downsampling stage
        """
        super().__init__()
        self.n_cells = n_cells
        n_nodes = len(cell_config)

        # Stem: initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        # Build cells with downsampling every ceil(n_cells/3) cells
        self.cells = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        c_in = init_channels
        c_out = init_channels
        downsample_interval = max(1, n_cells // 3)

        for i in range(n_cells):
            # Downsample before certain cells (but not the first)
            if i > 0 and i % downsample_interval == 0:
                c_out = min(c_out * channel_multiplier, 512)  # cap at 512
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                    nn.BatchNorm2d(c_out),
                ))
                # After downsampling, cell input channels = c_out
                cell = Cell(n_nodes, c_out, c_out, cell_config)
            else:
                self.downsamples.append(None)
                cell = Cell(n_nodes, c_in, c_out, cell_config)

            self.cells.append(cell)
            c_in = c_out

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_out, num_classes)

    def forward(self, x):
        x = self.stem(x)

        for i, cell in enumerate(self.cells):
            if self.downsamples[i] is not None:
                x = self.downsamples[i](x)
            x = cell(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# SEARCH SPACE SAMPLING
def sample_cell_config(n_nodes, rng=None):
    """
    Sample a random cell configuration.
    Returns list of (op_name, input_node_idx) of length n_nodes.
    """
    if rng is None:
        rng = random.Random()

    config = []
    for node_idx in range(n_nodes):
        op = rng.choice(OP_NAMES)
        # Each node can take input from node 0 (cell input) or any previous node
        input_idx = rng.randint(0, node_idx)
        config.append((op, input_idx))
    return config

def build_model_from_config(cell_config, in_channels, num_classes, n_cells, init_channels):
    """Build a CellNetwork from a cell configuration."""
    return CellNetwork(
        in_channels=in_channels,
        num_classes=num_classes,
        cell_config=cell_config,
        n_cells=n_cells,
        init_channels=init_channels,
        channel_multiplier=2,
    )

# TRAINING-FREE PROXY SCORES
def compute_naswot_score(model, dataloader, device, n_batches=1):
    """
    Compute the NASWOT (Neural Architecture Search Without Training) score.
    Based on Mellor et al. (ICML 2021): measures the correlation of activations
    at ReLU layers using a mini-batch, computing the log-determinant of the
    binary kernel matrix K where K[i,j] = hamming_similarity of activation patterns.
    """
    model.eval()
    model.to(device)

    # Collect ReLU activation patterns
    activations = {}
    hooks = []

    def get_hook(name):
        def hook_fn(module, input, output):
            # Store binary activation pattern (which neurons are active)
            activations[name] = (output > 0).float().detach()
        return hook_fn

    # Register hooks on ReLU layers
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(get_hook(f'relu_{idx}')))
            idx += 1

    if idx == 0:
        # No ReLU layers found, return 0
        for h in hooks:
            h.remove()
        return 0.0

    # Forward pass with one mini-batch
    try:
        batch_data = None
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch_data = batch[0]
            else:
                batch_data = batch
            break

        if batch_data is None:
            for h in hooks:
                h.remove()
            return 0.0

        batch_data = batch_data.to(device)
        with torch.no_grad():
            model(batch_data)
    except Exception:
        for h in hooks:
            h.remove()
        return 0.0

    # Compute kernel matrix from binary activation patterns
    # Flatten each sample's activation pattern across all ReLU layers
    batch_size = batch_data.shape[0]
    patterns = []
    for key in sorted(activations.keys()):
        act = activations[key]
        # Flatten spatial dims: (batch, channels, h, w) -> (batch, channels*h*w)
        act_flat = act.view(batch_size, -1)
        patterns.append(act_flat)

    if not patterns:
        for h in hooks:
            h.remove()
        return 0.0

    # Concatenate all patterns: (batch, total_neurons)
    all_patterns = torch.cat(patterns, dim=1).cpu().numpy()

    # Compute kernel matrix: K[i,j] = fraction of neurons with same activation
    # Using hamming similarity
    n = all_patterns.shape[0]
    d = all_patterns.shape[1]

    if d == 0:
        for h in hooks:
            h.remove()
        return 0.0

    # K[i,j] = (number of matching bits) / d
    K = np.dot(all_patterns, all_patterns.T) + np.dot(1 - all_patterns, (1 - all_patterns).T)
    K = K / d

    # Score = log(det(K)) — higher means more diverse representations
    # Add small diagonal for numerical stability
    K += np.eye(n) * 1e-5

    try:
        sign, logdet = np.linalg.slogdet(K)
        score = logdet if sign > 0 else -float('inf')
    except np.linalg.LinAlgError:
        score = -float('inf')

    # Clean up hooks
    for h in hooks:
        h.remove()

    return float(score)

def compute_synflow_score(model, dataloader, device):
    """
    Compute the SynFlow (Synaptic Flow) score.
    Based on Tanaka et al. (NeurIPS 2020): a data-agnostic pruning metric that
    computes the product of parameter magnitudes along each path in the network.
    The score is sum of (param * param.grad) after a single forward pass with all-ones input.
    """
    model.eval()
    model.to(device)

    # Make all parameters positive (absolute value) for path computation
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param.data)
        param.data = torch.abs(param.data)

    # Create all-ones input matching the batch shape
    try:
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                sample = batch[0]
            else:
                sample = batch
            break
        ones_input = torch.ones_like(sample[:1]).to(device)
    except Exception:
        # Restore signs and return 0
        for name, param in model.named_parameters():
            if name in signs:
                param.data = param.data * signs[name]
        return 0.0

    # Forward pass
    model.zero_grad()
    try:
        # Enable gradients temporarily
        for param in model.parameters():
            param.requires_grad_(True)

        output = model(ones_input)
        # Sum all outputs as scalar loss
        loss = output.sum()
        loss.backward()
    except Exception:
        for name, param in model.named_parameters():
            if name in signs:
                param.data = param.data * signs[name]
        return 0.0

    # SynFlow score: sum of |param * grad|
    score = 0.0
    for param in model.parameters():
        if param.grad is not None:
            score += (param.data * param.grad.data).abs().sum().item()

    # Restore original parameter signs
    for name, param in model.named_parameters():
        if name in signs:
            param.data = param.data * signs[name]
        param.requires_grad_(True)

    model.zero_grad()
    return float(score) if not math.isinf(score) and not math.isnan(score) else 0.0

def compute_param_count(model):
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# DIVERSITY ISLANDS
def classify_architecture(cell_config, n_cells, init_channels):
    """
    Classify an architecture into a diversity island based on its properties.
    Islands:
      - 'efficient': low param count (small channels, fewer cells, separable convs)
      - 'deep': more cells, deeper networks
      - 'residual': has skip connections
    """
    has_skip = any(op == 'skip' for op, _ in cell_config)
    has_sep = any('sep' in op for op, _ in cell_config)
    has_pool = any('pool' in op for op, _ in cell_config)
    heavy_ops = sum(1 for op, _ in cell_config if op in ('conv5x5', 'sep5x5'))

    # Heuristic classification
    if has_skip and has_sep:
        return 'residual'
    elif n_cells >= 5 or heavy_ops >= 2:
        return 'deep'
    else:
        return 'efficient'

def assign_island(cell_config, n_cells, init_channels):
    """Assign an architecture to a diversity island."""
    return classify_architecture(cell_config, n_cells, init_channels)


# COMBINED SCORING
def compute_combined_score(naswot, synflow, naswot_weight=0.5):
    """
    Combine NASWOT and SynFlow scores into a single ranking score.
    Both are normalized relative to the candidate pool before combining.
    """
    return naswot_weight * naswot + (1 - naswot_weight) * synflow

def normalize_scores(scores):
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-10:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]