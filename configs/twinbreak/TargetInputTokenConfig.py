from enum import Enum


class TargetInputTokenConfig(str, Enum):
    """
     This class holds the option that determines which `input` tokens are considered when taking mean over the token activations to find safety critical neurons.
    """
    # Keep all the input tokens of the prompt
    ALL = 'all'
    # Only keep the input tokens defined in target_input_tokens_custom_pos are considerd. Usually used to set it to -6 to include the INST token
    CUSTOM = 'custom'
    # Only the last input token is considered
    LAST = 'last'
    # Input tokens are not considered (`n_out_tokens_to_collect` must be greater than 1)
    NONE = 'none'
