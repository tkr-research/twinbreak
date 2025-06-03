from enum import Enum

class TokenMeanConfig(str, Enum):
    """
        This class defines how TwinBreak takes the mean over tokens
    """
    # Takes average over activations of all the considered token positions when identifying safety critical neurons
    ALL = 'all'
    # Takes average over activations of the `top_n_critical` (defined in config) token positions when identifying safety critical neurons
    CRITICAL = 'critical'
