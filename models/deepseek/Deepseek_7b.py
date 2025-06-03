from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.deepseek.AbstractDeepseekModel import AbstractDeepseekModel


class Deepseek_7b(AbstractDeepseekModel):
    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        super(Deepseek_7b, self).__init__(precision, hugging_face_token, store_model_disk_path)

    def get_huggingface_path(self) -> str:
        return "deepseek-ai/deepseek-llm-7b-chat"
