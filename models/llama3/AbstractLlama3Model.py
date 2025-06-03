from abc import ABCMeta
from typing import Tuple

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.AbstractModel import AbstractModel


class AbstractLlama3Model(AbstractModel, metaclass=ABCMeta):
    """
    This is common behavior for all LLaMA 3 models
    """

    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(AbstractLlama3Model, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_chat_template(self) -> str:
        return """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def initialize_tokenizer(self) -> None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def initialize_model(self) -> None:
        # Remove randomness already here
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

    def extract_prompt_and_response(self, output: str) -> Tuple[str, str]:
        response = output.split('assistant', 1)[-1].strip()
        prompt = output.split('assistant')[0].split('user', 1)[-1].strip()
        return prompt, response
