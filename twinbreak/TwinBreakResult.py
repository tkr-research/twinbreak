from typing import Dict

from configs.twinbreak.TargetModuleConfig import TargetModuleConfig
from configs.twinbreak.TwinBreakConfig import TwinBreakConfig
from models.AbstractModel import AbstractModel
from twinbreak.TwinBreakResultBucket import TwinBreakResultBucket


class TwinBreakResult:
    def __init__(self, model: AbstractModel, config: TwinBreakConfig, input_token_length: int, num_layer: int):
        gate_dim = model.get_layer_list()[0].mlp.gate_proj.out_features
        down_dim = model.get_layer_list()[0].mlp.down_proj.out_features
        up_dim = model.get_layer_list()[0].mlp.up_proj.out_features
        attn_dim = model.get_layer_list()[0].self_attn.o_proj.out_features

        self.result_buckets: Dict[TargetModuleConfig, TwinBreakResultBucket] = {}

        if TargetModuleConfig.GATE in config.target_modules:
            self.result_buckets[TargetModuleConfig.GATE] = TwinBreakResultBucket(config, gate_dim, input_token_length,
                                                                                 num_layer)
        if TargetModuleConfig.UP in config.target_modules:
            self.result_buckets[TargetModuleConfig.UP] = TwinBreakResultBucket(config, up_dim, input_token_length,
                                                                               num_layer)
        if TargetModuleConfig.DOWN in config.target_modules:
            self.result_buckets[TargetModuleConfig.DOWN] = TwinBreakResultBucket(config, down_dim, input_token_length,
                                                                                 num_layer)
        if TargetModuleConfig.ATTN in config.target_modules:
            self.result_buckets[TargetModuleConfig.ATTN] = TwinBreakResultBucket(config, attn_dim, input_token_length,
                                                                                 num_layer)

    def __getitem__(self, item: TargetModuleConfig):
        return self.result_buckets[item]

    def reset_activation_caches(self) -> None:
        for _, result in self.result_buckets.items():
            result.reset_activation_cache()

    def reset_safety_parameters(self) -> None:
        for _, result in self.result_buckets.items():
            result.reset_safety_parameters()

    def compute_utility_parameters(self) -> None:
        for _, result in self.result_buckets.items():
            result.compute_utility_parameters()

    def compute_safety_parameters(self, pruning_iteration: int, number_of_prompts: int) -> None:
        for _, result in self.result_buckets.items():
            result.compute_safety_parameters(pruning_iteration, number_of_prompts)

    def save_results(self, result_folder_path: str, pruning_iteration: int) -> None:
        for identifier, result in self.result_buckets.items():
            result.save_results(identifier, result_folder_path, pruning_iteration)

    def load_results(self, result_folder_path: str, pruning_iteration: int) -> None:
        for identifier, result in self.result_buckets.items():
            result.load_results(identifier, result_folder_path, pruning_iteration)
