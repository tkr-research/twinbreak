from enum import Enum


class BatchModeConfig(str, Enum):
    """
        This setting controls in what batches we calculate the safety parameters
    """
    # Devide the 100 TwinPrompt into random batches of size batch_size. We feed each batch individually and identify 1/number_of_batches safety parameters from each batch iterativelly in each iteration.
    FULLITER = 'fulliter'
    # Divides prompt pairs into batches. In each pruning iteration, we only process one batch. So if the number_of_batches is larger than the pruning iterations, batches will not be considered.
    BATCHITER = 'batchiter'
    # Default: No batching - we take the full TwinPrompt dataset in one batch to the model. Find safety parameters based on all the prompt pairs.
    NONE = 'none'
