from abc import ABCMeta
from typing import Tuple

from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.AbstractModel import AbstractModel


class AbstractLlama2Model(AbstractModel, metaclass=ABCMeta):
    """
    This is common behavior for all LLaMA 2 models
    """

    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(AbstractLlama2Model, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_chat_template(self) -> str:
        return "[INST] {instruction} [/INST] "

    def initialize_tokenizer(self) -> None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def extract_prompt_and_response(self, output: str) -> Tuple[str, str]:
        response = output.split('[/INST]')[-1].strip()
        prompt = output.split('[/INST]')[0].split('[INST]')[-1].strip()
        return prompt, response
