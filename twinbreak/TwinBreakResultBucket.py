from math import ceil
from typing import List

import torch

from configs.twinbreak.BatchModeConfig import BatchModeConfig
from configs.twinbreak.TargetInputTokenConfig import TargetInputTokenConfig
from configs.twinbreak.TargetModuleConfig import TargetModuleConfig
from configs.twinbreak.TokenMeanConfig import TokenMeanConfig
from configs.twinbreak.TwinBreakConfig import TwinBreakConfig


class TwinBreakResultBucket:
    def __init__(self, config: TwinBreakConfig, dim: int, input_token_length: int, num_layers: int):
        self.__config: TwinBreakConfig = config
        # Dimensions of the layer
        self.layer_output_activation_dimensions: int = dim
        # Numer of tokens
        self.input_token_length: int = input_token_length
        self.num_tokens: int = self.input_token_length + self.__config.n_out_tokens_to_collect - 1
        # Numer of layers
        self.num_layers: int = num_layers
        # Activation cache for one set of prompts that we monitor, e.g., harmless prompts - stored temporarily for one evaluation run
        self.activation_cache_one: torch.Tensor = torch.zeros(
            (self.num_tokens, num_layers, self.layer_output_activation_dimensions),
            dtype=torch.float64)
        # Activation cache for the second set of prompts that we monitor, e.g., harmful prompts - stored temporarily for one evaluation run
        self.activation_cache_two: torch.Tensor = torch.zeros(
            (self.num_tokens, num_layers, self.layer_output_activation_dimensions),
            dtype=torch.float64)
        # Parameters that we identified
        self.safety_parameters: List[None | torch.Tensor] = [None for _ in range(num_layers)]
        # Utility parameters that we identified
        self.utility_parameters: List[None | torch.Tensor] = [None for _ in range(num_layers)]

    def get_safety_parameters(self) -> List[None | torch.Tensor]:
        return self.safety_parameters

    def get_activation_cache_one(self) -> torch.Tensor:
        return self.activation_cache_one

    def get_activation_cache_two(self) -> torch.Tensor:
        return self.activation_cache_two

    def reset_safety_parameters(self) -> None:
        self.safety_parameters = [None for _ in range(self.num_layers)]

    def reset_activation_cache(self) -> None:
        self.activation_cache_one: torch.Tensor = torch.zeros(
            (self.num_tokens, self.num_layers, self.layer_output_activation_dimensions),
            dtype=torch.float64)
        self.activation_cache_two: torch.Tensor = torch.zeros(
            (self.num_tokens, self.num_layers, self.layer_output_activation_dimensions),
            dtype=torch.float64)

    def compute_utility_parameters(self) -> None:
        number_of_parameters_to_identify = int(
            self.layer_output_activation_dimensions * self.__config.utility_parameter_rate)
        # Hint: We now iterate over all layers (also the ones not selected), which has no effect as those layers do not have any hook registered.
        for layer_idx in range(self.num_layers):
            activation_differences_mean = self.__calculate_mean_over_activation_differences(
                layer_idx)
            critical_parameters = torch.topk(activation_differences_mean,
                                             k=number_of_parameters_to_identify).indices.int()
            self.utility_parameters[layer_idx] = critical_parameters

    def __calculate_mean_over_activation_differences(self, layer_idx: int) -> torch.Tensor:
        # Calculate difference between the two caches, e.g., harmless and harmful prompts
        activation_differences = torch.abs(self.activation_cache_two - self.activation_cache_one)[:, layer_idx, :]
        # We produced activations for input_token_length input tokens
        in_activations = activation_differences[0:self.input_token_length]
        # And for some output tokens (probably 1)
        out_activations = activation_differences[self.input_token_length:]

        # Only keep the input tokens as defined in the config
        target_input_token = self.__config.target_input_tokens
        if target_input_token == TargetInputTokenConfig.CUSTOM:
            # Only keep the input tokens after INST with the configured position
            in_activations = in_activations[self.__config.target_input_tokens_custom_pos:, :]
            activation_differences = torch.cat((in_activations, out_activations))
        elif target_input_token == TargetInputTokenConfig.ALL:
            # We keep all inputs and append outputs
            activation_differences = torch.cat((in_activations, out_activations))
        elif target_input_token == TargetInputTokenConfig.LAST:
            # We only append the last input with all outputs
            activation_differences = torch.cat((in_activations[-1, :].unsqueeze(0), out_activations))
        elif target_input_token == TargetInputTokenConfig.NONE:
            # We take not inputs and only outputs.
            activation_differences = out_activations
        else:
            raise Exception(f"Unknown target input token: {target_input_token}")

        # Take mean over the entire input/out token positions
        mean_over_tokens = self.__config.mean_over_tokens

        if mean_over_tokens == TokenMeanConfig.ALL:
            activation_differences_mean = activation_differences.mean(0)
        # Take mean over the top-`n-critical` input/out tokens with the largest gap
        elif mean_over_tokens == TokenMeanConfig.CRITICAL:
            critical_tokens = torch.topk(torch.norm(activation_differences, dim=-1),
                                         k=min(self.__config.top_n_critical, activation_differences.shape[0])).indices
            activation_differences_mean = activation_differences[critical_tokens, :].mean(0)
        else:
            raise Exception(f"Unknown mean over tokens: {mean_over_tokens}")

        return activation_differences_mean

    def compute_safety_parameters(self, pruning_iteration: int, number_of_prompts: int) -> None:
        # First get the current pruning rate by iteration
        if pruning_iteration == 0:
            pruning_rate = self.__config.initial_parameter_pruning_rate
        else:
            pruning_rate = self.__config.parameter_pruning_step
        number_of_parameters_to_identify = int(self.layer_output_activation_dimensions * pruning_rate)
        # Hint: We now iterate over all layers (also the ones not selected), which has no effect as those layers do not have any hook registered.
        for layer_idx in range(self.num_layers):
            activation_differences_mean = self.__calculate_mean_over_activation_differences(layer_idx)

            if self.__config.batch_mode == BatchModeConfig.BATCHITER or self.__config.batch_mode == BatchModeConfig.NONE:
                # We processed either one batch or the whole datset as one batch.
                # We want to have the gap from all the data in the activation cache.
                critical_parameters = torch.topk(activation_differences_mean,
                                                 k=number_of_parameters_to_identify).indices.int()
            elif self.__config.batch_mode == BatchModeConfig.FULLITER:
                # We want to compute the gap individually for each batch defined by batch_mode_batch_size
                number_of_parameters_to_identify_in_batch = ceil(
                    (
                            self.__config.batch_mode_batch_size / number_of_prompts) * number_of_parameters_to_identify)
                critical_parameters = torch.topk(activation_differences_mean,
                                                 k=number_of_parameters_to_identify_in_batch).indices.int()
            else:
                raise Exception("Batch mode not supported.")

            # Maybe we want to exclude the utility parameters
            if self.__config.exclude_utility_nodes:
                critical_parameters = torch.tensor(
                    list(set(critical_parameters.tolist()) - set(self.utility_parameters[layer_idx].tolist()))
                    , dtype=torch.int32)
            else:
                critical_parameters = torch.tensor(list(critical_parameters.tolist()), dtype=torch.int32)

            if self.safety_parameters[layer_idx] is None:
                self.safety_parameters[layer_idx] = critical_parameters
            else:
                # By default we add up all the parameters
                self.safety_parameters[layer_idx] = torch.cat([self.safety_parameters[layer_idx], critical_parameters])

    def save_results(self, identifier: TargetModuleConfig, result_folder_path: str, pruning_iteration: int) -> None:
        torch.save(self.utility_parameters,
                   f'{result_folder_path}utility_parameter_pruning_iteration_{pruning_iteration}_{identifier.value}.pt')
        torch.save(self.safety_parameters,
                   f'{result_folder_path}safety_parameter_pruning_iteration_{pruning_iteration}_{identifier.value}.pt')

    def load_results(self, identifier: TargetModuleConfig, result_folder_path: str, pruning_iteration: int) -> None:
        self.utility_parameters = torch.load(
            f'{result_folder_path}utility_parameter_pruning_iteration_{pruning_iteration}_{identifier.value}.pt')
        self.safety_parameters = torch.load(
            f'{result_folder_path}safety_parameter_pruning_iteration_{pruning_iteration}_{identifier.value}.pt')
