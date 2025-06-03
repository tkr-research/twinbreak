from abc import ABCMeta
from typing import Tuple

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.AbstractModel import AbstractModel


class AbstractGemma3Model(AbstractModel, metaclass=ABCMeta):
    """
    This is common behavior for all Gemma3 models
    """

    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(AbstractGemma3Model, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_chat_template(self) -> str:
        return """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

    def initialize_model(self) -> None:
        # Remove randomness already here
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def initialize_tokenizer(self) -> None:
        self.tokenizer.padding_side = 'left'

    def extract_prompt_and_response(self, output: str) -> Tuple[str, str]:
        response = output.split('model', 1)[-1].strip()
        prompt = output.split('model')[0].split('user', 1)[-1].strip()
        return prompt, response
