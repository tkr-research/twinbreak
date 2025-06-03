import os
from typing import List

import yaml
from dotenv import load_dotenv

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from configs.twinbreak.TwinBreakConfig import TwinBreakConfig
from dataset.DatasetsIdentifier import DatasetIdentifier
from dataset.UtilityDatasetIdentifier import UtilityDatasetIdentifier
from helper.FileHandler import FileHandler
from models.ModelConfig import ModelConfig
from models.ModelIdentifier import ModelIdentifier


class ExperimentConfig:
    """
    This class holds the parameters for an experiment in this artifact.
    """

    def __init__(self, experiment_config_file_path: str, experiment_identifier: str):
        with open(experiment_config_file_path) as f:
            config = yaml.safe_load(f)

        # The output folder for results
        experiment_folder_name = os.path.dirname(__file__)
        # Load local environment variables
        env_file = f"{experiment_folder_name}/../../.env"
        load_dotenv(env_file)
        # Define the output folder
        self.output_folder = FileHandler.ensure_dir_exists(experiment_folder_name,
                                                           f'../../results/{experiment_identifier}')

        # The access token for HuggingFace
        # Replace placeholder with environment variable
        hugging_face_token: str = os.getenv("HF_TOKEN")
        # Raise an error if the token is missing
        if not hugging_face_token:
            raise RuntimeError("Missing Hugging Face token. Set the HF_TOKEN environment variable.")
        # The seed for the randomness, see RandomHandler
        self.seed: int = int(config['seed'])
        # The precision of the LLM. See PrecisionConfig enum for values
        precision: PrecisionConfig = PrecisionConfig(config['precision'])
        # How many data are fed during inference to your GPU. Adjust based on your GPU memory size.
        self.batch_size: int = int(config['batch_size'])
        # File path to the JSON config file for TwinBreak
        twinbreak_config_file_path: str = experiment_folder_name + '/' + str(config['twinbreak_config_file_path'])
        # Location with enough storage space for Huggingface to download and cache your models
        store_model_disk_path: str = os.getenv("STORE_MODEL_DISK_PATH")
        # Raise an error if the token is missing
        if not store_model_disk_path:
            raise RuntimeError("Missing stroe model disk path. Set the STORE_MODEL_DISK_PATH environment variable.")
        self.twinbreak_config: TwinBreakConfig = TwinBreakConfig(twinbreak_config_file_path, self.batch_size,
                                                                 self.output_folder, precision, hugging_face_token,
                                                                 store_model_disk_path)
        # The model we use in this experiment
        model_identifier: ModelIdentifier = ModelIdentifier(config['model_identifier'])
        self.model_config: ModelConfig = ModelConfig(model_identifier, precision, hugging_face_token,
                                                     store_model_disk_path)
        # List of all harmful datasets used for safety evaluation. See DatasetIdentifier enum for values.
        self.safety_evaluation_datasets: List[DatasetIdentifier] = [DatasetIdentifier(ds) for ds in
                                                                    config["safety_evaluation_datasets"]]
        # List of all utility datasets or benchmarks used for utility evaluation. See UtilityDatasetIdentifier enum for values.
        self.utility_evaluation_datasets: List[UtilityDatasetIdentifier] = [UtilityDatasetIdentifier(ds) for ds in
                                                                            config["utility_evaluation_datasets"]]

    def __str__(self):
        return (
            f"ExperimentConfig(\n"
            f"  output_folder={self.output_folder},\n"
            f"  seed={self.seed},\n"
            f"  batch_size={self.batch_size},\n"
            f"  model_config=\n{str(self.model_config).replace(chr(10), chr(10) + '    ')},\n"
            f"  safety_evaluation_datasets={[ds.value for ds in self.safety_evaluation_datasets]},\n"
            f"  utility_evaluation_datasets={[ds.value for ds in self.utility_evaluation_datasets]},\n"
            f"  twinbreak_config=\n{str(self.twinbreak_config).replace(chr(10), chr(10) + '    ')}\n"
            f")"
        )
