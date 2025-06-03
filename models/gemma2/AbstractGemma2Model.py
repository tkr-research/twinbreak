from abc import ABCMeta
from typing import Tuple

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.AbstractModel import AbstractModel


class AbstractGemma2Model(AbstractModel, metaclass=ABCMeta):
    """
    This is common behavior for all Gemma2 models
    """

    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(AbstractGemma2Model, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_chat_template(self) -> str:
        return """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

    def initialize_tokenizer(self) -> None:
        self.tokenizer.padding_side = 'left'
        # As explained in https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
        self.tokenizer.pad_token = '<|endoftext|>'
        self.tokenizer.pad_token_id = 151643

    def extract_prompt_and_response(self, output: str) -> Tuple[str, str]:
        response = output.split('model', 1)[-1].strip()
        prompt = output.split('model')[0].split('user', 1)[-1].strip()
        return prompt, response
