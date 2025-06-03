from enum import Enum

class DatasetIdentifier(str, Enum):
    """
    This class holds the names of the dataset files for easier access.
    """
    # The Advbench dataset
    ADVBENCH = "advbench"
    # The TwinPrompt dataset consists of 100 samples from the Harmbench dataset (half of Harmbench)
    # The Harmbench samples are extended manually with a benign twin prompt.
    TWINPROMPT = "twinprompt"
    # The second half (100) prompts from harmbench
    HARMBENCH_VALIDATION = "harmbench_validation"
    # The Jailbreakbench dataset
    JAILBREAKBENCH = "jailbreakbench"
    # The Strongreject dataset
    STRONGREJECT = "strongreject"
    # A dataset with benign and malicious prompts (similar to TwinPrompt) but with NON similar prompts.
    # Used for abluation study
    ABLATION_NON_SIMILAR = "ablation_non_similarity"
