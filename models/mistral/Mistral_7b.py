from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.mistral.AbstractMistralModel import AbstractMistralModel


class Mistral_7b(AbstractMistralModel):
    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(Mistral_7b, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_huggingface_path(self) -> str:
        return "mistralai/Mistral-7B-Instruct-v0.2"
