from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.gemma2.AbstractGemma2Model import AbstractGemma2Model


class Gemma2_27b(AbstractGemma2Model):
    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(Gemma2_27b, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_huggingface_path(self) -> str:
        return "google/gemma-2-27b-it"
