from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.qwen25.AbstractQwen25Model import AbstractQwen25Model


class Qwen25_32b(AbstractQwen25Model):
    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(Qwen25_32b, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_huggingface_path(self) -> str:
        return "Qwen/Qwen2.5-32B-Instruct"
