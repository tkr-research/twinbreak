import torch

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig


class DTypeConfigFactory:

    @staticmethod
    def create(precision: PrecisionConfig) -> torch.dtype:
        if precision == PrecisionConfig.BIT16:
            dtype = torch.float16
        elif precision == PrecisionConfig.BIT32:
            dtype = torch.float32
        elif precision == PrecisionConfig.BIT64:
            dtype = torch.float64
        else:
            raise Exception(f"Unknown PrecisionConfig: {precision}")
        return dtype
