import os

from configs.experiments.ExperimentConfig import ExperimentConfig
from experiments.ExperimentExecutor import ExperimentExecutor
from experiments.ExperimentVisualizer import ExperimentVisualizer

if __name__ == '__main__':
    experiment_config_file_path = '../configs/experiments/experiment_3_2.yaml'
    experiment_file_name = os.path.splitext(os.path.basename(__file__))[0]
    # If you want to run multiple runs of one experiment, you can change the name in the line below, e.g., add _1 to it.
    experiment_file_name = f'{experiment_file_name}'
    experiment_config = ExperimentConfig(experiment_config_file_path, experiment_file_name)
    executor = ExperimentExecutor(experiment_config)
    executor.run()
    # Print a summary of this experiment
    model_name = "Qwen 2.5 (3b) [24]"
    utility_paper_locations = {
        "hellaswag": "Table 8 (Utility)"
        }
    safety_paper_locations = {
        "strongreject": "Table 8 (ASR)"
        }
    ExperimentVisualizer.generate_summary(model_name, utility_paper_locations,
                                          safety_paper_locations, True)
