from configs.experiments.dtype.PrecisionConfig import PrecisionConfig
from models.ModelIdentifier import ModelIdentifier


class ModelConfig:
    def __init__(self, identifier: ModelIdentifier, precision: PrecisionConfig, hugging_face_token: str,
                 store_model_disk_path: str):
        # The model we use in this experiment
        self.identifier: ModelIdentifier = identifier
        # What is the precision of the LLM used
        self.precision: PrecisionConfig = precision
        # Token used to access hugging face
        self.hugging_face_token: str = hugging_face_token
        # The path where the model can be stored on the dis
        self.store_model_disk_path: str = store_model_disk_path

    def __str__(self):
        return (
            f"ModelConfig(\n"
            f"  identifier={self.identifier.value},\n"
            f"  precision={self.precision.value},\n"
            f"  hugging_face_token={'<hidden>' if self.hugging_face_token else '<none>'},\n"
            f"  store_model_disk_path={self.store_model_disk_path}\n"
            f")"
        )
