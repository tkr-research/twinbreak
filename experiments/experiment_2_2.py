import os

from configs.experiments.ExperimentConfig import ExperimentConfig
from experiments.ExperimentExecutor import ExperimentExecutor
from experiments.ExperimentVisualizer import ExperimentVisualizer

if __name__ == '__main__':
    experiment_config_file_path = '../configs/experiments/experiment_2_2.yaml'
    experiment_file_name = os.path.splitext(os.path.basename(__file__))[0]
    # If you want to run multiple runs of one experiment, you can change the name in the line below, e.g., add _1 to it.
    experiment_file_name = f'{experiment_file_name}'
    experiment_config = ExperimentConfig(experiment_config_file_path, experiment_file_name)
    executor = ExperimentExecutor(experiment_config)
    executor.run()
    # Print a summary of this experiment
    model_name = "Gemma 2 (9b) [19]"
    utility_paper_locations = {
        "hellaswag": "Table 15", "rte": "Table 16", "openbookqa": "Table 17", "arc_challenge": "Table 18",
        "winogrande": "Table 19"
        }
    safety_paper_locations = {
        "harmbench_validation": "Table 2 (Val)", "advbench": "Table 3 (Val)", "jailbreakbench": "Table 4 (Val)",
        "strongreject": "Table 5 (Val)"
        }
    ExperimentVisualizer.generate_summary(model_name, utility_paper_locations,
                                          safety_paper_locations, False)
