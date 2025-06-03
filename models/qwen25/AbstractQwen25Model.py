from abc import ABCMeta
from typing import Tuple

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.AbstractModel import AbstractModel


class AbstractQwen25Model(AbstractModel, metaclass=ABCMeta):
    """
    This is common behavior for all Qwen25 models
    """

    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(AbstractQwen25Model, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_chat_template(self) -> str:
        return """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

    def initialize_tokenizer(self) -> None:
        self.tokenizer.padding_side = 'left'

    def extract_prompt_and_response(self, output: str) -> Tuple[str, str]:
        response = output.split('assistant', 1)[-1].strip()
        prompt = output.split('assistant')[0].split('user', 1)[-1].strip()
        return prompt, response
