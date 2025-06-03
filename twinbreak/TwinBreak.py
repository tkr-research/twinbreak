from math import ceil
from typing import Callable, List, Tuple

import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import GenerationConfig

from configs.twinbreak.BatchModeConfig import BatchModeConfig
from configs.twinbreak.TargetModuleConfig import TargetModuleConfig
from configs.twinbreak.TwinBreakConfig import TwinBreakConfig
from dataset.DatasetBucket import DatasetBucket
from dataset.DatasetBucketFactory import DatasetBucketFactory
from helper.LoggingHandler import LoggingHandler
from helper.RandomHandler import RandomHandler
from models.AbstractModel import AbstractModel
from twinbreak.TwinBreakResult import TwinBreakResult


class TwinBreak:
    def __init__(self, config: TwinBreakConfig):
        self._config: TwinBreakConfig = config
        # A running variable that keeps track of tokens we generated with the model (as we are using hooks)
        self._num_generated_tokens: int = 0

    def run(self, model: AbstractModel) -> Tuple[List[Tuple[TargetModuleConfig, nn.Linear, int, int]], Callable[
        [nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]]:
        dataset: DatasetBucket = self.__load_dataset(model)
        results: TwinBreakResult = self.__create_result_objects(model, dataset)
        targeted_layer_list = self.__get_targeted_layer_list(model)
        head_hook = self.__register_hook_to_count_generated_tokens(model)
        LoggingHandler.log_and_print_prepend_timestamps(
            "TIMEMEASUREMENT ATTACK_UTILITY: Identification of utility parameters...")
        self.__compute_utility_parameters(results, dataset, model, targeted_layer_list)
        LoggingHandler.log_and_print_prepend_timestamps(
            "TIMEMEASUREMENT ATTACK_UTILITY: Identification of utility parameters done.")
        LoggingHandler.log_and_print_prepend_timestamps(
            "TIMEMEASUREMENT ATTACK_SAFETY: Identification of safety parameters...")
        self.__process_twin_dataset_and_compute_safety_parameters(results, dataset, model, targeted_layer_list)
        LoggingHandler.log_and_print_prepend_timestamps(
            "TIMEMEASUREMENT ATTACK_SAFETY: Identification of safety parameters done.")
        return targeted_layer_list, head_hook

    def __load_dataset(self, model: AbstractModel) -> DatasetBucket:
        LoggingHandler.log_and_print_prepend_timestamps("Load TwinPrompt dataset...")
        dataset = DatasetBucketFactory.create(self._config.dataset_identifier)
        LoggingHandler.log_and_print_prepend_timestamps("Load TwinPrompt dataset done.")
        LoggingHandler.log_and_print_prepend_timestamps("Tokenize TwinPrompt dataset...")
        dataset.tokenize(model)
        LoggingHandler.log_and_print_prepend_timestamps("Tokenize TwinPrompt dataset done.")
        return dataset

    def __create_result_objects(self, model: AbstractModel, dataset: DatasetBucket) -> TwinBreakResult:
        LoggingHandler.log_and_print_prepend_timestamps(
            "Create bucket for activation caching and TwinBreak results (utility and safety parameters)...")
        results = TwinBreakResult(model, self._config, dataset.get_input_token_length(),
                                  model.get_hidden_layer_num())
        LoggingHandler.log_and_print_prepend_timestamps(
            "Create bucket for activation caching and TwinBreak results (utility and safety parameters) done.")
        return results

    def __get_targeted_layer_list(self, model: AbstractModel) -> List[Tuple[TargetModuleConfig, nn.Linear, int, int]]:
        """
        Returns a list of all the LLM modules that are to be targeted.
        The list contains four-element tuples: `(module_name: TargetModuleConfig, module: nn.Linear, layer_idx: int, out_dim: int)`
        """

        LoggingHandler.log_and_print_prepend_timestamps(
            "Create list of all LLM modules (layers) that are targeted for pruning and with this for activation tracking...")
        module_list: List[Tuple[
            TargetModuleConfig, nn.Linear, int, int]] = []

        start, end = self._config.target_layers
        if end < 0:
            end = model.get_hidden_layer_num() + end
        layer_indices = list(range(start, end))

        for layer_idx in layer_indices:
            LoggingHandler.log_and_print_prepend_timestamps(f"Handling layer with index {layer_idx}.")
            if TargetModuleConfig.GATE in self._config.target_modules:
                LoggingHandler.log_and_print_prepend_timestamps("Added GATE layers to the list.")
                module_list.append((TargetModuleConfig.GATE, model.get_layer_list()[layer_idx].mlp.gate_proj,
                                    layer_idx,
                                    model.get_layer_list()[layer_idx].mlp.gate_proj.out_features))

            if TargetModuleConfig.DOWN in self._config.target_modules:
                LoggingHandler.log_and_print_prepend_timestamps("Added DOWN layers to the list.")
                module_list.append((TargetModuleConfig.DOWN, model.get_layer_list()[layer_idx].mlp.down_proj,
                                    layer_idx,
                                    model.get_layer_list()[layer_idx].mlp.down_proj.out_features))

            if TargetModuleConfig.UP in self._config.target_modules:
                LoggingHandler.log_and_print_prepend_timestamps("Added UP layers to the list.")
                module_list.append((TargetModuleConfig.UP, model.get_layer_list()[layer_idx].mlp.up_proj,
                                    layer_idx,
                                    model.get_layer_list()[layer_idx].mlp.up_proj.out_features))

            if TargetModuleConfig.ATTN in self._config.target_modules:
                LoggingHandler.log_and_print_prepend_timestamps("Added ATTN layers to the list.")
                module_list.append((TargetModuleConfig.ATTN, model.get_layer_list()[layer_idx].self_attn.o_proj,
                                    layer_idx,
                                    model.get_layer_list()[layer_idx].self_attn.o_proj.out_features))
        LoggingHandler.log_and_print_prepend_timestamps(
            "Create list of all LLM modules (layers) that are targeted for pruning and with this for activation tracking done.")
        return module_list

    def __register_hook_to_count_generated_tokens(self, model: AbstractModel) -> Callable[
        [nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        LoggingHandler.log_and_print_prepend_timestamps(f"Register forward hook to count generated tokens...")
        head_hook = model.get_model_head().register_forward_hook(self.__head_hook())
        LoggingHandler.log_and_print_prepend_timestamps(f"Register forward hook to count generated tokens done.")
        return head_hook

    def __head_hook(self) -> Callable[[nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """
        Hooked to the last layer of the target model.
        """

        def hook_fn(module: nn.Module, input_data: Tuple[Tensor], output: Tensor) -> None:
            self._num_generated_tokens += 1

        return hook_fn

    def __compute_utility_parameters(self, result: TwinBreakResult, dataset: DatasetBucket,
                                     model: AbstractModel, targeted_layer_list: List[
            Tuple[TargetModuleConfig, nn.Linear, int, int]]) -> None:
        # Reset the current activation cache
        result.reset_activation_caches()

        # Split the harmless prompts into two sets of equal size
        LoggingHandler.log_and_print_prepend_timestamps("Generating harmless prompt paris...")
        half = dataset.get_size() // 2
        harmless_prompts_tokenized_1 = dataset.get_harmless_prompts_tokenized()[0:half]
        harmless_mask_tokenized_1 = dataset.get_harmless_mask_tokenized()[0:half]
        harmless_prompts_tokenized_2 = dataset.get_harmless_prompts_tokenized()[half:]
        harmless_mask_tokenized_2 = dataset.get_harmless_mask_tokenized()[half:]
        LoggingHandler.log_and_print_prepend_timestamps("Generating harmless prompt paris done.")

        LoggingHandler.log_and_print_prepend_timestamps("Inference with harmless prompt pairs...")

        # Run the data through the model (could be batched but does not make a difference)
        # Feed the first set of harmless prompts
        self.__process_data_and_track_activations(result, model, targeted_layer_list,
                                                  harmless_prompts_tokenized_1,
                                                  harmless_mask_tokenized_1, True, False,
                                                  dataset.get_input_token_length(), True)

        # Feed the second set of harmless prompts
        self.__process_data_and_track_activations(result, model, targeted_layer_list,
                                                  harmless_prompts_tokenized_2,
                                                  harmless_mask_tokenized_2, False, False,
                                                  dataset.get_input_token_length(), True)

        LoggingHandler.log_and_print_prepend_timestamps("Inference with harmless prompt pairs done.")

        # Compute utility parameters
        LoggingHandler.log_and_print_prepend_timestamps("Compute utility parameters by comparing activations...")
        result.compute_utility_parameters()
        LoggingHandler.log_and_print_prepend_timestamps("Compute utility parameters by comparing activations done.")

        # Reset the activation cache again
        result.reset_activation_caches()

        # Reset the number of generated tokens
        self._num_generated_tokens = 0

    def __process_data_and_track_activations(self, result: TwinBreakResult, model: AbstractModel,
                                             targeted_layer_list: List[Tuple[TargetModuleConfig, nn.Linear, int, int]],
                                             prompts_tokenized: torch.Tensor, masks_tokenized: torch.Tensor,
                                             first_twin_prompt: bool, prune_model: bool, input_token_length: int,
                                             store_activations_in_cache: bool) -> \
            List[str]:
        # We first reset the number of tokens that we generated
        self._num_generated_tokens = 0
        # We register a hook that tracks activations and also prunes the model during inference if necessary
        LoggingHandler.log_and_print_prepend_timestamps("Registering hooks for activation tracking and / or pruning...")
        hooks = []
        for module_config, layer, layer_idx, out_features in targeted_layer_list:
            # Select the correct activation cache from the respective result object
            if first_twin_prompt:
                activation_cache = result[module_config].get_activation_cache_one()
            else:
                activation_cache = result[module_config].get_activation_cache_two()
            safety_parameters = None
            if prune_model:
                safety_parameters = result[module_config].get_safety_parameters()
            hook_fn = self._save_and_prune_hook(layer_idx, prune_model,
                                                safety_parameters,
                                                store_activations_in_cache, activation_cache
                                                , len(prompts_tokenized),
                                                input_token_length)
            hooks.append(layer.register_forward_hook(hook_fn))

        LoggingHandler.log_and_print_prepend_timestamps(
            "Registering hooks for activation tracking and / or pruning done.")
        LoggingHandler.log_and_print_prepend_timestamps("Inference...")
        outputs = self._process_data(model, prompts_tokenized, masks_tokenized, self._config.n_out_tokens_to_collect)
        LoggingHandler.log_and_print_prepend_timestamps("Inference done.")

        # We remove the hooks again
        LoggingHandler.log_and_print_prepend_timestamps("Removing hooks for activation tracking and / or pruning...")
        for hook in hooks:
            hook.remove()

        LoggingHandler.log_and_print_prepend_timestamps("Removing hooks for activation tracking and / or pruning done.")
        return outputs

    def _save_and_prune_hook(self,
                             layer_idx: int,
                             prune: bool,
                             safety_parameters: List[None | torch.Tensor] | None,
                             store_activations_in_cache: bool,
                             activation_cache: torch.Tensor | None, number_of_prompts: int | None,
                             input_token_length: int | None) -> Callable:
        if store_activations_in_cache:
            assert activation_cache is not None, "activation_cache should not be none if you store_activations_in_cache."
        else:
            assert activation_cache is None, "You do not need an activation_cache if you do not store_activations_in_cache."
        if store_activations_in_cache:
            assert number_of_prompts is not None, "number_of_prompts should not be none if you store_activations_in_cache."
        else:
            assert number_of_prompts is None, "You do not need number_of_prompts if you do not store_activations_in_cache."
        if store_activations_in_cache:
            assert input_token_length is not None, "input_token_length should not be none if you store_activations_in_cache."
        else:
            assert input_token_length is None, "You do not need input_token_length if you do not store_activations_in_cache."

        if prune:
            assert safety_parameters is not None, "safety_parameters should not be none if you prune."
        else:
            assert safety_parameters is None, "You do not need safety_parameters if you do not prune."

        def hook(module, inp, out):
            # Sometimes model hold the activations in a tuple with one element, sometimes not
            # Afterward, the activation will have the following shape: [num_samples, num_input_tokens, dim]
            if isinstance(out, tuple):
                activations = out[0]
            else:
                activations = out

            # Maybe we prune
            if prune:
                if self._num_generated_tokens < self._config.n_out_tokens_to_apply_pruning:
                    if safety_parameters[layer_idx] is not None:
                        activations[:, :, safety_parameters[layer_idx]] = 0.0

            # Activation Collection if necesasry
            if store_activations_in_cache and self._num_generated_tokens < self._config.n_out_tokens_to_collect:
                # How many samples are stored in the activations?
                if self._config.batch_mode == BatchModeConfig.BATCHITER:
                    # As we only processed one batch in this pruning iteration, the average is computed by dividing by this value
                    dividing_factor = self._config.batch_mode_batch_size
                else:
                    # For fulliter and none, all data have been processed
                    dividing_factor = number_of_prompts

                if activations.shape[1] == 1:
                    # We land here if subsequent output tokens after the first one were generated.
                    # Here we do the average by sum and div.
                    activation_cache[
                        input_token_length + self._num_generated_tokens - 1, layer_idx] += torch.div(
                        activations.to(activation_cache.device).sum(0).squeeze(0), dividing_factor)
                else:
                    # We land here when the first output token was generated
                    # Here we do the average by sum and div
                    activation_cache[0:input_token_length, layer_idx] += torch.div(
                        activations.to(activation_cache.device).sum(0), dividing_factor)

            # Retrun the activations - they changed, if we pruned
            if isinstance(out, tuple):
                return activations, *out[1:]
            else:
                return activations

        return hook

    def __process_twin_dataset_and_compute_safety_parameters(self, result: TwinBreakResult, dataset: DatasetBucket,
                                                             model: AbstractModel, targeted_layer_list: List[
            Tuple[TargetModuleConfig, nn.Linear, int, int]]) -> None:
        # Count in which batch we are if we use batchiter
        batch_counter_for_batch_mode_config_batchiter: int | None = None
        # How many batches can be constructed with the batch_mode_batch_size for batchiter
        total_number_of_batches_available_for_batch_mode_config_batchiter: int | None = None

        for pruning_iteration in range(self._config.pruning_iterations):
            LoggingHandler.log_and_print_prepend_timestamps(f"Pruning iteration {pruning_iteration}...")
            # Reset the current activation cache
            result.reset_activation_caches()

            if self._config.batch_mode == BatchModeConfig.BATCHITER:
                if batch_counter_for_batch_mode_config_batchiter is None or batch_counter_for_batch_mode_config_batchiter == total_number_of_batches_available_for_batch_mode_config_batchiter:
                    # Calculate a new permutation to randomly build new batches
                    RandomHandler.renew_randomness_with_extra_value(pruning_iteration)
                    permutation = torch.randperm(dataset.get_size())
                    # Permutate the data
                    permuted_harmless_prompts_tokenized = dataset.get_harmless_prompts_tokenized()[permutation]
                    permuted_harmless_mask_tokenized = dataset.get_harmless_mask_tokenized()[permutation]
                    permuted_harmful_prompts_tokenized = dataset.get_harmful_prompts_tokenized()[permutation]
                    permuted_harmful_mask_tokenized = dataset.get_harmful_mask_tokenized()[permutation]
                    # Calculate how many batches we have
                    total_number_of_batches_available_for_batch_mode_config_batchiter = ceil(
                        dataset.get_size() / self._config.batch_mode_batch_size)
                    # Initialize the counter
                    if batch_counter_for_batch_mode_config_batchiter is None:
                        batch_counter_for_batch_mode_config_batchiter = 0
                # Now build the current batches
                start_idx = batch_counter_for_batch_mode_config_batchiter * self._config.batch_mode_batch_size
                end_idx = (batch_counter_for_batch_mode_config_batchiter + 1) * self._config.batch_mode_batch_size

                batch_harmless_prompts_tokenized = permuted_harmless_prompts_tokenized[start_idx:end_idx]
                batch_harmless_mask_tokenized = permuted_harmless_mask_tokenized[start_idx:end_idx]
                batch_harmful_prompts_tokenized = permuted_harmful_prompts_tokenized[start_idx:end_idx]
                batch_harmful_mask_tokenized = permuted_harmful_mask_tokenized[start_idx:end_idx]

                # Run the data through the model (could be batched but does not make a difference)
                # Feed the harmless prompts
                self.__process_data_and_track_activations(result, model, targeted_layer_list,
                                                          batch_harmless_prompts_tokenized,
                                                          batch_harmless_mask_tokenized, True, True,
                                                          dataset.get_input_token_length(), True)

                # Feed the harmful prompts
                self.__process_data_and_track_activations(result, model, targeted_layer_list,
                                                          batch_harmful_prompts_tokenized,
                                                          batch_harmful_mask_tokenized, False, True,
                                                          dataset.get_input_token_length(), True)

                # Increase the counter
                batch_counter_for_batch_mode_config_batchiter += 1

            elif self._config.batch_mode == BatchModeConfig.FULLITER:
                # Calculate a new permutation to randomly build new batches
                RandomHandler.renew_randomness_with_extra_value(pruning_iteration)
                permutation = torch.randperm(dataset.get_size())
                # Permutate the data
                permuted_harmless_prompts_tokenized = dataset.get_harmless_prompts_tokenized()[permutation]
                permuted_harmless_mask_tokenized = dataset.get_harmless_mask_tokenized()[permutation]
                permuted_harmful_prompts_tokenized = dataset.get_harmful_prompts_tokenized()[permutation]
                permuted_harmful_mask_tokenized = dataset.get_harmful_mask_tokenized()[permutation]

                for i in tqdm(range(0, dataset.get_size(), self._config.batch_mode_batch_size),
                              desc="Processing batches", unit="batch"):
                    batch_harmless_prompts_tokenized = permuted_harmless_prompts_tokenized[
                                                       i:i + self._config.batch_mode_batch_size]
                    batch_harmless_mask_tokenized = permuted_harmless_mask_tokenized[
                                                    i:i + self._config.batch_mode_batch_size]
                    batch_harmful_prompts_tokenized = permuted_harmful_prompts_tokenized[
                                                      i:i + self._config.batch_mode_batch_size]
                    batch_harmful_mask_tokenized = permuted_harmful_mask_tokenized[
                                                   i:i + self._config.batch_mode_batch_size]
                    # Collect activations (with pruning on)
                    for j in range(0, len(batch_harmless_prompts_tokenized), self._config.batch_size):
                        # Feed the first set of harmless prompts
                        self.__process_data_and_track_activations(result, model, targeted_layer_list,
                                                                  batch_harmless_prompts_tokenized[
                                                                  j:j + self._config.batch_size],
                                                                  batch_harmless_mask_tokenized[
                                                                  j:j + self._config.batch_size], True, True,
                                                                  dataset.get_input_token_length(), True)

                        # Feed the second set of harmful prompts
                        self.__process_data_and_track_activations(result, model, targeted_layer_list,
                                                                  batch_harmful_prompts_tokenized[
                                                                  j:j + self._config.batch_size],
                                                                  batch_harmful_mask_tokenized[
                                                                  j:j + self._config.batch_size], False, True,
                                                                  dataset.get_input_token_length(), True)


            elif self._config.batch_mode == BatchModeConfig.NONE:
                for j in tqdm(range(0, dataset.get_size(), self._config.batch_size), desc="Processing batches",
                              unit="batch"):
                    # Feed the first set of harmless prompts
                    self.__process_data_and_track_activations(result, model, targeted_layer_list,
                                                              dataset.get_harmless_prompts_tokenized()[
                                                              j:j + self._config.batch_size],
                                                              dataset.get_harmless_mask_tokenized()[
                                                              j:j + self._config.batch_size], True, True,
                                                              dataset.get_input_token_length(), True)

                    # Feed the second set of harmful prompts
                    self.__process_data_and_track_activations(result, model, targeted_layer_list,
                                                              dataset.get_harmful_prompts_tokenized()[
                                                              j:j + self._config.batch_size],
                                                              dataset.get_harmful_mask_tokenized()[
                                                              j:j + self._config.batch_size], False, True,
                                                              dataset.get_input_token_length(), True)

            else:
                raise Exception("Invalid batch mode")

            # Compute the safety parameters based on the batch
            LoggingHandler.log_and_print_prepend_timestamps(f"Compare activations and compute safety parameters...")
            result.compute_safety_parameters(pruning_iteration, dataset.get_size())
            LoggingHandler.log_and_print_prepend_timestamps(f"Compare activations and compute safety parameters done.")

            # Reset the activation cache again
            result.reset_activation_caches()

            LoggingHandler.log_and_print_prepend_timestamps(f"Save results to disk...")
            result.save_results(self._config.output_folder, pruning_iteration)
            LoggingHandler.log_and_print_prepend_timestamps(f"Save results to disk done.")

            # Clear pruning indices if aggregation is turned off
            if not self._config.aggregate_pruning:
                result.reset_safety_parameters()

            LoggingHandler.log_and_print_prepend_timestamps(f"Pruning iteration {pruning_iteration} done.")
        # Reset the activation cache again
        result.reset_activation_caches()

    def _process_data(self, model: AbstractModel, prompts_tokenized: torch.Tensor, masks_tokenized: torch.Tensor,
                      number_of_tokens_to_generate: int) -> \
            List[str]:
        # We do inference with the data
        # Ensure deterministic outputs and set the maximum number of output tokens
        generation_config = GenerationConfig(max_new_tokens=number_of_tokens_to_generate,
                                             do_sample=False,
                                             num_beams=1)
        generation_config.pad_token_id = model.tokenizer.pad_token_id
        out = model.model.generate(input_ids=prompts_tokenized.to(model.model.device),
                                   attention_mask=masks_tokenized.to(model.model.device),
                                   generation_config=generation_config)
        outputs = model.tokenizer.batch_decode(out, skip_special_tokens=True)

        return outputs
