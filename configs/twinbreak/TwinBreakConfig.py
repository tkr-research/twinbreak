from typing import List, Tuple

import yaml

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from configs.twinbreak.BatchModeConfig import BatchModeConfig
from configs.twinbreak.TargetInputTokenConfig import TargetInputTokenConfig
from configs.twinbreak.TargetModuleConfig import TargetModuleConfig
from configs.twinbreak.TokenMeanConfig import TokenMeanConfig
from dataset.DatasetsIdentifier import DatasetIdentifier
from helper.FileHandler import FileHandler


class TwinBreakConfig:
    """
    This class holds the parameters of TwinBreak. It reads them from a JSON file,  holds the parameters locally and provides getters to them.
    """

    def __init__(self, config_file_path: str, batch_size: int, output_folder: str, precision: PrecisionConfig,
                 hugging_face_token: str,
                 store_model_disk_path: str):
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config_json: str = config
        # Whether to identify and exclude utility neurons when pruning
        self.exclude_utility_nodes: bool = bool(config['exclude_utility_nodes'])
        # The rate of top utility parameters to be excluded from pruning
        self.utility_parameter_rate: float = float(config['utility_parameter_rate'])
        # Range of decoder block layers that undergo pruning. Format int_int.
        target_layers = config['target_layers']
        if '_' in target_layers:
            start, end = [int(num) for num in target_layers.split('_')]
        else:
            assert False, "'target_layers' must have the format 'int_int'"
        self.target_layers: Tuple[int, int] = (start, end)
        # Which MLP components should be pruned. See TargetModuleConfig for values.
        target_modules = config['target_modules']
        self.target_modules: List[TargetModuleConfig] = []
        assert isinstance(target_modules,
                          list), "'target_modules' must be a list with some or all of the elements 'gate', 'down', 'up', 'attn'"
        for target_module in target_modules:
            self.target_modules.append(TargetModuleConfig(target_module))
        # Which input tokens are considered when taking mean over the token activations to find safety critical neurons. See TargetInputTokenConfig for values.
        self.target_input_tokens: TargetInputTokenConfig = TargetInputTokenConfig(config['target_input_tokens'])
        # If target_input_tokens is custom this value specifies the last n positions of the input tokens whose activations are considered
        self.target_input_tokens_custom_pos: int = int(config['target_input_tokens_custom_pos'])
        assert self.target_input_tokens_custom_pos < 0, (
            "target_input_tokens_custom_pos must be < 0 to indicate the laste -n tokens")
        # The number of output tokens to generate (`N`). Hint: The number of output tokens to consider when identifying safety critical neurons will be `N - 1` as for example the last six tokens are used for evaluation, which includes one output token less than we generated.
        self.n_out_tokens_to_collect: int = int(config['n_out_tokens_to_collect'])
        # Safety check
        if self.target_input_tokens == TargetInputTokenConfig.NONE:
            assert self.n_out_tokens_to_collect > 1, (
                "When target_input_tokens is set to `none`, `n_out_tokens_to_collect` must be greater than 1")
        # If we take the mean over all or only the top_n_critical tokens
        self.mean_over_tokens: TokenMeanConfig = TokenMeanConfig(config['mean_over_tokens'])
        # Number of tokens with the largest L2 norm activations among all the input and output tokens that are considered when taking average over token activations
        self.top_n_critical: int = int(config['top_n_critical'])
        # How many output tokens to generate under pruned model
        self.n_out_tokens_to_apply_pruning: int = int(config['n_out_tokens_to_apply_pruning'])
        # Whether to aggregate pruned parameters during pruning iterations (True) or prune based on the parameters identified at the current pruning iterations (False).
        self.aggregate_pruning: bool = bool(config['aggregate_pruning'])
        # Number of pruning iterations
        self.pruning_iterations = int(config['pruning_iterations'])
        # In what batches do we compute the safety parameters. See BatchModeConfig for values.
        self.batch_mode: BatchModeConfig = BatchModeConfig(config['batch_mode'])
        # Batch size for safety parameter identification - necessary is batch_mode is not NONE
        self.batch_mode_batch_size = int(config['batch_mode_batch_size'])
        # Initial rate of top safety critical neurons to be pruned
        self.initial_parameter_pruning_rate = float(config['initial_parameter_pruning_rate'])
        # The rate of top safety critical neurons to be pruned in iterations > 1
        self.parameter_pruning_step = float(config['parameter_pruning_step'])
        # The dataset we use for TwinBreak - which usually is TwinPrompt
        self.dataset_identifier: DatasetIdentifier = DatasetIdentifier(config['dataset_identifier'])
        # Indicates if we use the complete training dataset or not. TwinPrompt contains 100 samples
        self.training_size: int = config['training_size']
        self.training_size = int(config['training_size'])
        assert self.training_size <= 100, "Training size must be smaller than 100"
        assert self.training_size > 0, "Training size must be greater than 0"
        # Batch size used for inference - usually set by the experiment due to hardware
        self.batch_size: int = batch_size
        # Subfolder for outputs
        self.output_folder: str = FileHandler.ensure_dir_exists(output_folder, 'twinbreak')
        # Settings for LlamaGuard Evaluation
        self.precision: PrecisionConfig = precision
        self.hugging_face_token: str = hugging_face_token
        self.store_model_disk_path: str = store_model_disk_path

    def __str__(self):
        return (
            f"TwinBreakConfig(\n"
            f"  exclude_utility_nodes={self.exclude_utility_nodes},\n"
            f"  utility_parameter_rate={self.utility_parameter_rate},\n"
            f"  target_layers={self.target_layers},\n"
            f"  target_modules={[m.value for m in self.target_modules]},\n"
            f"  target_input_tokens={self.target_input_tokens.value},\n"
            f"  target_input_tokens_custom_pos={self.target_input_tokens_custom_pos},\n"
            f"  n_out_tokens_to_collect={self.n_out_tokens_to_collect},\n"
            f"  mean_over_tokens={self.mean_over_tokens.value},\n"
            f"  top_n_critical={self.top_n_critical},\n"
            f"  n_out_tokens_to_apply_pruning={self.n_out_tokens_to_apply_pruning},\n"
            f"  aggregate_pruning={self.aggregate_pruning},\n"
            f"  pruning_iterations={self.pruning_iterations},\n"
            f"  batch_mode={self.batch_mode.value},\n"
            f"  batch_mode_batch_size={self.batch_mode_batch_size},\n"
            f"  initial_parameter_pruning_rate={self.initial_parameter_pruning_rate},\n"
            f"  parameter_pruning_step={self.parameter_pruning_step},\n"
            f"  dataset_identifier={self.dataset_identifier.value},\n"
            f"  training_size={self.training_size},\n"
            f"  batch_size={self.batch_size},\n"
            f"  output_folder={self.output_folder},\n"
            f"  precision={self.precision.value},\n"
            f"  store_model_disk_path={self.store_model_disk_path}\n"
            f")"
        )
