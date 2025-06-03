import os
import random

import numpy as np
import torch


class RandomHandler:
    ACTUAL_SEED: int = 42
    DEBUG: bool = False

    @staticmethod
    def renew_seed_of_random_methods(seed: int, debug: bool) -> None:
        RandomHandler.ACTUAL_SEED = seed
        RandomHandler.DEBUG = debug
        RandomHandler.renew_randomness_with_extra_value(0)

    @staticmethod
    def renew_randomness_with_extra_value(extra: int) -> None:
        seed = RandomHandler.ACTUAL_SEED + extra
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # From Docs https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html#torch.cuda.manual_seed:
        # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        use_deterministic_algorithms = not RandomHandler.DEBUG
        torch.use_deterministic_algorithms(use_deterministic_algorithms)
