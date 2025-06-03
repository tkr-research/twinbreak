from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.gemma3.AbstractGemma3Model import AbstractGemma3Model


class Gemma3_1b(AbstractGemma3Model):
    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(Gemma3_1b, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_huggingface_path(self) -> str:
        return "google/gemma-3-1b-it"
