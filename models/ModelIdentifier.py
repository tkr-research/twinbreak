from enum import Enum

class ModelIdentifier(str, Enum):
    """
    Class to hold our model identifiers.
    """
    LLAMA2_7b = 'llama2_7b'
    LAAMA2_13B = 'llama2_13b'
    LLAMA2_70B = 'llama2_70b'
    LLAMA3_1_8b = 'llama3_1_8b'
    LLAMA3_3_70b = 'llama3_3_70b'
    QWEN2_3B = 'qwen2_3b'
    QWEN2_7B = 'qwen2_7b'
    QWEN2_14B = 'qwen2_14b'
    QWEN2_32B = 'qwen2_32b'
    QWEN2_72B = 'qwen2_72b'
    GEMMA2_2B = 'gemma2_2b'
    GEMMA2_9B = 'gemma2_9b'
    GEMMA2_27B = 'gemma2_27b'
    Gemma3_1b = 'gemma3_1b'
    MISTRAL_7b = 'mistral_7b'
    DEEPSEEK_7b = 'deepseek_7b'
