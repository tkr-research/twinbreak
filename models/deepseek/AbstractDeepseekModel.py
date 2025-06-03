from abc import ABCMeta
from typing import Tuple

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.AbstractModel import AbstractModel


class AbstractDeepseekModel(AbstractModel, metaclass=ABCMeta):
    """
    This is common behavior for all DeepSeek models
    """

    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(AbstractDeepseekModel, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_chat_template(self) -> str:
        return """User: {instruction}

Assistant:"""

    def initialize_tokenizer(self) -> None:
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def initialize_model(self) -> None:
        # Remove randomness already here
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

    def extract_prompt_and_response(self, output: str) -> Tuple[str, str]:
        response = output.split('Assistant:', 1)[-1].strip()
        prompt = output.split('Assistant:')[0].split('User: ', 1)[-1].strip()
        return prompt, response
