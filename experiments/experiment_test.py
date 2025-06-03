import os

from configs.experiments.ExperimentConfig import ExperimentConfig
from experiments.ExperimentExecutor import ExperimentExecutor

if __name__ == '__main__':
    experiment_config_file_path = '../configs/experiments/experiment_default_settings.yaml'
    experiment_file_name = os.path.splitext(os.path.basename(__file__))[0]
    # Get the full path of the current file
    # If you want to run multiple runs of one experiment, you can change the name in the line below, e.g., add _1 to it.
    experiment_file_name = f'{experiment_file_name}'
    experiment_config = ExperimentConfig(experiment_config_file_path, experiment_file_name)
    executor = ExperimentExecutor(experiment_config)
    print(f"The system is set up to run experiments!")
