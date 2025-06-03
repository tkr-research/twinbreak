from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.llama2.AbstractLlama2Model import AbstractLlama2Model


class Llama2_13b(AbstractLlama2Model):
    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(Llama2_13b, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_huggingface_path(self) -> str:
        return "meta-llama/Llama-2-13b-chat-hf"
