from enum import Enum


class TargetModuleConfig(str, Enum):
    """
    This class is an enum that defines the possible target modules (or sub layers) that we aim to prune in TwinBreak.
    """
    # Target the MLP gate layer
    GATE = 'gate'
    # Target the MLP down layer
    DOWN = 'down'
    # Target the MLP up layer
    UP = 'up'
    # Target self-attention layer
    ATTN = 'attn'
