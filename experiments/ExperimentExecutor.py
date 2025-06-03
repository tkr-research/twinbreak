from configs.experiments.ExperimentConfig import ExperimentConfig
from helper.FileHandler import FileHandler
from helper.LoggingHandler import LoggingHandler
from helper.RandomHandler import RandomHandler
from models.AbstractModel import AbstractModel
from models.ModelFactory import ModelFactory
from twinbreak.TwinBreakAndEval import TwinBreakAndEval


class ExperimentExecutor:
    """
    This class executes an experiment
    """

    def __init__(self, experiment_config: ExperimentConfig):
        # Change this to true if you want to debug the code
        self.experiment_config = experiment_config

    def run(self):
        self.__create_output_folders()
        self.__initialize_logger()
        self.__log_configurations()
        self.__load_model()
        self.__initialize_random()
        self.__attack_and_eval_model()

    def __create_output_folders(self) -> None:
        print(f'Creating output folders...')
        self.folder_main = FileHandler.ensure_dir_exists(self.experiment_config.output_folder)
        self.folder_log = FileHandler.ensure_dir_exists(self.folder_main, 'log')
        self.folder_data = FileHandler.ensure_dir_exists(self.folder_main, 'data')
        print(f'Creating output folders done.')

    def __initialize_logger(self) -> None:
        print("Initialize logger...")
        LoggingHandler.init_log_file(self.folder_log)
        LoggingHandler.log_and_print_prepend_timestamps(f'Logger initialized')

    def __log_configurations(self) -> None:
        LoggingHandler.log_and_print_prepend_timestamps("Log parameter for this experiment...")
        LoggingHandler.log_and_print_prepend_timestamps(str(self.experiment_config))
        FileHandler.write_to_file(f'{self.folder_main}/config.txt', str(self.experiment_config))
        LoggingHandler.log_and_print_prepend_timestamps("Log parameter for this experiment done.")

    def __load_model(self) -> None:
        LoggingHandler.log_and_print_prepend_timestamps("Create the model instance...")
        self.model: AbstractModel = ModelFactory.create(self.experiment_config.model_config)
        LoggingHandler.log_and_print_prepend_timestamps("Create the model instance done.")

    def __initialize_random(self) -> None:
        LoggingHandler.log_and_print_prepend_timestamps("Initialize Randomness.")
        # Renew the seeds
        seed = self.experiment_config.seed
        LoggingHandler.log_and_print_prepend_timestamps(f'Seeds set to {seed}')
        RandomHandler.renew_seed_of_random_methods(seed, True)
        LoggingHandler.log_and_print_prepend_timestamps("Randomness initialized.")

    def __attack_and_eval_model(self) -> None:
        LoggingHandler.log_and_print_prepend_timestamps("Attack the model with TwinBreak and evaluate results...")
        twin_break = TwinBreakAndEval(self.experiment_config.twinbreak_config)
        twin_break.run_and_eval(self.model, self.experiment_config.safety_evaluation_datasets,
                                self.experiment_config.utility_evaluation_datasets)
        LoggingHandler.log_and_print_prepend_timestamps("Attack the model with TwinBreak and evaluate results done.")
