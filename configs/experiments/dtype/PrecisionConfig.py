from enum import Enum

class PrecisionConfig(str, Enum):
    """
    Numerical precision used for computations. Options include `float16`, `float32`, and `float64`
    """
    BIT8 = 'bit8'
    BIT16 = 'bit16'
    BIT32 = 'bit32'
    BIT64 = 'bit64'
