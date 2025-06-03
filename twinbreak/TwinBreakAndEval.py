import json
from typing import Dict, List, Tuple

import torch
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from torch import nn

from configs.twinbreak.TargetModuleConfig import TargetModuleConfig
from configs.twinbreak.TwinBreakConfig import TwinBreakConfig
from dataset.DatasetBucket import DatasetBucket
from dataset.DatasetBucketFactory import DatasetBucketFactory
from dataset.DatasetsIdentifier import DatasetIdentifier
from dataset.UtilityDatasetIdentifier import UtilityDatasetIdentifier
from helper.LlamaGuardHandler import LlamaGuardHandler
from helper.LoggingHandler import LoggingHandler
from helper.StrongRejectHandler import StrongRejectHandler
from models.AbstractModel import AbstractModel
from twinbreak.TwinBreak import TwinBreak


class TwinBreakAndEval(TwinBreak):
    def __init__(self, config: TwinBreakConfig):
        super().__init__(config)

    def run_and_eval(self, model: AbstractModel, safety_task_list: List[DatasetIdentifier],
                     utility_task_list: List[UtilityDatasetIdentifier]) -> None:
        # Run the TwinBreak Attack
        LoggingHandler.log_and_print_prepend_timestamps("TIMEMEASUREMENT ATTACK: Attack the model with TwinBreak...")
        targeted_layer_list, head_hook = self.run(model)
        LoggingHandler.log_and_print_prepend_timestamps("TIMEMEASUREMENT ATTACK: Attack the model with TwinBreak done.")
        LoggingHandler.log_and_print_prepend_timestamps("Evaluate TwinBreak performance...")
        # First we generate the results on the safety benchmarks
        LoggingHandler.log_and_print_prepend_timestamps("Generate outputs from the pruned model...")
        self.__generate_safety_task_responses(model, safety_task_list, targeted_layer_list)
        LoggingHandler.log_and_print_prepend_timestamps("Generate outputs from the pruned model done.")
        # Remove the hook as we do not need it for utility any more
        LoggingHandler.log_and_print_prepend_timestamps(
            "Remove forward hook, as output tokens do not need to be counted any more.")
        head_hook.remove()
        # Now we evaluate utility first
        LoggingHandler.log_and_print_prepend_timestamps("Evaluate utility on defined benchmarks...")
        self.__evaluate_utility(model, utility_task_list, targeted_layer_list)
        LoggingHandler.log_and_print_prepend_timestamps("Evaluate utility on defined benchmarks done.")
        # We then evaluate the generated results on the safety benchmarks
        LoggingHandler.log_and_print_prepend_timestamps("Evaluate safety of generated outputs...")
        self.__evaluate_safety_of_generated_safety_task_responses(safety_task_list)
        LoggingHandler.log_and_print_prepend_timestamps("Evaluate safety of generated outputs done.")
        LoggingHandler.log_and_print_prepend_timestamps("Evaluate TwinBreak performance done.")

    def __evaluate_utility(self, model: AbstractModel, utility_task_list: List[UtilityDatasetIdentifier],
                           targeted_layer_list: List[
                               Tuple[TargetModuleConfig, nn.Linear, int, int]]) -> None:

        # We first reset the number of tokens that we generated
        self._num_generated_tokens = 0
        # We prepend a None pruning iteration for the unchanged model
        pruning_iterations = [None]
        for pruning_iteration in range(self._config.pruning_iterations):
            pruning_iterations.append(pruning_iteration)
        for pruning_iteration in pruning_iterations:
            LoggingHandler.log_and_print_prepend_timestamps(
                f"Evaluate utility for pruning iteration {pruning_iteration}...")
            assert self._num_generated_tokens == 0, "Head hook must be removed."
            hooks = []
            if pruning_iteration is not None:
                LoggingHandler.log_and_print_prepend_timestamps(f"Load safety parameters and register pruning hook.")
                # If we use a pruning iteration, we need to prune via hooks, so we register them
                for module_config, layer, layer_idx, out_features in targeted_layer_list:
                    # Get the safety parameters for the round.
                    safety_parameters = torch.load(
                        f'{self._config.output_folder}safety_parameter_pruning_iteration_{pruning_iteration}_{module_config.value}.pt')
                    hook_fn = self._save_and_prune_hook(layer_idx,
                                                        True,
                                                        safety_parameters,
                                                        False, None, None,
                                                        None)
                    hooks.append(layer.register_forward_hook(hook_fn))
            else:
                LoggingHandler.log_and_print_prepend_timestamps(
                    f"No safety parameters are used and no pruning, as we evaluate the unpruned model.")

            # Inference
            LoggingHandler.log_and_print_prepend_timestamps(f"Inference for all the defined utility tasks...")
            utility_task_list_str = [e.value for e in utility_task_list]
            hflm = HFLM(pretrained=model.model, tokenizer=model.tokenizer, batch_size=self._config.batch_size)
            # 200 is the number of samples we use from each utility benchmark due to the size of them.
            results = simple_evaluate(model=hflm, tasks=utility_task_list_str, num_fewshot=0,
                                      batch_size=self._config.batch_size,
                                      limit=200)
            LoggingHandler.log_and_print_prepend_timestamps(f"Inference for all the defined utility tasks done.")
            # Log the results
            for challenge, result in results['results'].items():
                LoggingHandler.log_and_print_prepend_timestamps(
                    f'Results for utility benchmark {challenge} in TwinBreak pruning iteration {pruning_iteration}.')
                LoggingHandler.log_and_print_prepend_timestamps(f'{result}')
                with open(f"{self._config.output_folder}result_{challenge}_{pruning_iteration}.txt", "w") as outfile:
                    outfile.write(f'{result}\n')
                for metric, value in result.items():
                    if metric == "acc,none":
                        LoggingHandler.log_and_print_prepend_timestamps(
                            f'SUMMARY ACCURACY for utility benchmark {challenge} in pruning iteration {pruning_iteration}: {value}')
                        with open(f"{self._config.output_folder}acc_{challenge}_{pruning_iteration}.txt",
                                  "w") as outfile:
                            outfile.write(f'{value}\n')
                    elif metric == "acc_stderr,none":
                        LoggingHandler.log_and_print_prepend_timestamps(
                            f'ACCURACY STDERR for utility benchmark {challenge} in pruning iteration {pruning_iteration}: {value}')
                        with open(f"{self._config.output_folder}acc_std_{challenge}_{pruning_iteration}.txt",
                                  "w") as outfile:
                            outfile.write(f'{value}\n')

            # We remove the hooks again - if any
            LoggingHandler.log_and_print_prepend_timestamps(f"Removing pruning hook...")
            for hook in hooks:
                hook.remove()
            LoggingHandler.log_and_print_prepend_timestamps(f"Removing pruning hook done.")

            LoggingHandler.log_and_print_prepend_timestamps(
                f"Evaluate utility for pruning iteration {pruning_iteration} done.")

    def __generate_safety_task_responses(self, model: AbstractModel, safety_task_list: List[DatasetIdentifier],
                                         targeted_layer_list: List[
                                             Tuple[TargetModuleConfig, nn.Linear, int, int]]) -> None:

        # We prepend a None pruning iteration for the unchanged model
        pruning_iterations = [None]
        for pruning_iteration in range(self._config.pruning_iterations):
            pruning_iterations.append(pruning_iteration)

        datasets: Dict[DatasetIdentifier, DatasetBucket] = {}
        for pruning_iteration in pruning_iterations:
            LoggingHandler.log_and_print_prepend_timestamps(
                f'Start evaluating safety tasks for pruning iteration {pruning_iteration}')
            hooks = []
            if pruning_iteration is not None:
                # If we use a pruning iteration, we need to prune via hooks, so we register them
                for module_config, layer, layer_idx, out_features in targeted_layer_list:
                    # Get the safety parameters for the round.
                    safety_parameters = torch.load(
                        f'{self._config.output_folder}safety_parameter_pruning_iteration_{pruning_iteration}_{module_config.value}.pt')
                    hook_fn = self._save_and_prune_hook(layer_idx,
                                                        True,
                                                        safety_parameters,
                                                        False, None, None,
                                                        None)
                    hooks.append(layer.register_forward_hook(hook_fn))

            # Evaluate all the safety tasks on this model
            for safety_task in safety_task_list:
                LoggingHandler.log_and_print_prepend_timestamps(f'Start evaluating safety task {safety_task}')
                # Load the dataset
                if safety_task not in datasets.keys():
                    datasets[safety_task] = DatasetBucketFactory.create(safety_task)
                    datasets[safety_task].tokenize(model)
                dataset = datasets[safety_task]

                # List with prompt, response, and LLamaGuard Answer.
                output_safety_task: List[Tuple[str, str, str]] = []
                for j in range(0, dataset.get_size(), self._config.batch_size):
                    # We first reset the number of tokens that we generated
                    self._num_generated_tokens = 0
                    output = self._process_data(model, dataset.get_harmful_prompts_tokenized()[
                                                       j:j + self._config.batch_size],
                                                dataset.get_harmful_mask_tokenized()[
                                                j:j + self._config.batch_size], 500)
                    for o in output:
                        prompt, response = model.extract_prompt_and_response(o)
                        output_safety_task.append((prompt, response, ''))
                # Save task output to disk
                with open(f"{self._config.output_folder}safety_results_{safety_task}_{pruning_iteration}.json",
                          "w") as f:
                    json.dump(output_safety_task, f, indent=4)
                # Hint: We evaluate the results later (to save GPU memory)

            # We remove the hooks again - if any
            for hook in hooks:
                hook.remove()

    def __evaluate_safety_of_generated_safety_task_responses(self, safety_task_list: List[DatasetIdentifier]) -> None:
        # We prepend a None pruning iteration for the unchanged model
        pruning_iterations = [None]
        for pruning_iteration in range(self._config.pruning_iterations):
            pruning_iterations.append(pruning_iteration)

        LoggingHandler.log_and_print_prepend_timestamps(f'Load LlamaGuard model...')
        llamaguard = LlamaGuardHandler(self._config.hugging_face_token, self._config.store_model_disk_path,
                                       self._config.precision)
        LoggingHandler.log_and_print_prepend_timestamps(f'Load LlamaGuard model done.')
        strongreject = StrongRejectHandler()

        for safety_task in safety_task_list:
            LoggingHandler.log_and_print_prepend_timestamps(f'Evaluating safety benchmark {safety_task}...')
            cumulative_jailbreaks: List[float] | None = None
            num_prompts = 0
            for pruning_iteration in pruning_iterations:
                LoggingHandler.log_and_print_prepend_timestamps(
                    f'Evaluating safety benchmark {safety_task} in pruning iteration {pruning_iteration}...')
                # Load from file
                LoggingHandler.log_and_print_prepend_timestamps(f'Load generated (potentially) harmful prompt pairs...')
                with open(f"{self._config.output_folder}safety_results_{safety_task}_{pruning_iteration}.json",
                          "r") as f:
                    output_safety_task = json.load(f)
                LoggingHandler.log_and_print_prepend_timestamps(
                    f'Load generated (potentially) harmful prompt pairs done.')
                LoggingHandler.log_and_print_prepend_timestamps(f'Evaluate safety with auxiliary LLM...')
                if cumulative_jailbreaks is None:
                    num_prompts = len(output_safety_task)
                    cumulative_jailbreaks = [0 for _ in range(num_prompts)]
                if safety_task == DatasetIdentifier.STRONGREJECT:
                    scores_in_iteration, jailbreak_success_rate, output_safety_task = strongreject.evaluate(
                        output_safety_task)
                    cumulative_jailbreaks = [max(a, b) for a, b in zip(cumulative_jailbreaks, scores_in_iteration)]
                else:
                    jailbreaks_in_iteration, jailbreak_success_rate, output_safety_task = llamaguard.evaluate(
                        output_safety_task)
                    cumulative_jailbreaks = [float(a or b) for a, b in
                                             zip(cumulative_jailbreaks, jailbreaks_in_iteration)]
                LoggingHandler.log_and_print_prepend_timestamps(f'Evaluate safety with auxiliary LLM done.')
                # Dump results back
                LoggingHandler.log_and_print_prepend_timestamps(
                    f'Save generated (potentially) harmful prompt pairs with result to disk...')
                with open(f"{self._config.output_folder}safety_results_{safety_task}_{pruning_iteration}.json",
                          "w") as f:
                    json.dump(output_safety_task, f, indent=4)
                LoggingHandler.log_and_print_prepend_timestamps(
                    f'Save generated (potentially) harmful prompt pairs with result to disk done.')
                # Store jailbreak resutls
                with open(f"{self._config.output_folder}jailbreak_success_rate_{safety_task}_{pruning_iteration}.json",
                          "w") as outfile:
                    outfile.write(f'{jailbreak_success_rate}\n')
                LoggingHandler.log_and_print_prepend_timestamps(
                    f'Jailbreak success rate for safety benchmark {safety_task} in pruning iteration {pruning_iteration}: {jailbreak_success_rate}')

                jailbreak_percentage_cumulative = sum(cumulative_jailbreaks) / num_prompts
                with open(
                        f"{self._config.output_folder}jailbreak_success_rate_{safety_task}_{pruning_iteration}_cumulative.json",
                        "w") as outfile:
                    outfile.write(f'{jailbreak_percentage_cumulative}\n')
                LoggingHandler.log_and_print_prepend_timestamps(
                    f'jailbreak_success_rate_{safety_task}_{pruning_iteration}_cumulative: {jailbreak_percentage_cumulative}')
                LoggingHandler.log_and_print_prepend_timestamps(
                    f'SUMMARY JAILBREAK_SUCCESS_RATE cumulative for safety benchmark {safety_task} in pruning iteration {pruning_iteration}: {jailbreak_percentage_cumulative}')
