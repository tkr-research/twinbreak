from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.llama3.AbstractLlama3Model import AbstractLlama3Model


class Llama31_8b(AbstractLlama3Model):
    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(Llama31_8b, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_huggingface_path(self) -> str:
        return "meta-llama/Llama-3.1-8B-Instruct"
