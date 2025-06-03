from models.AbstractModel import AbstractModel
from models.ModelConfig import ModelConfig
from models.ModelIdentifier import ModelIdentifier
from models.deepseek.Deepseek_7b import Deepseek_7b
from models.gemma2.Gemma2_27b import Gemma2_27b
from models.gemma2.Gemma2_2b import Gemma2_2b
from models.gemma2.Gemma2_9b import Gemma2_9b
from models.gemma3.Gemma3_1b import Gemma3_1b
from models.llama2.Llama2_13b import Llama2_13b
from models.llama2.Llama2_70b import Llama2_70b
from models.llama2.Llama2_7b import Llama2_7b
from models.llama3.Llama31_8b import Llama31_8b
from models.llama3.Llama33_70b import Llama33_70b
from models.mistral.Mistral_7b import Mistral_7b
from models.qwen25.Qwen25_14b import Qwen25_14b
from models.qwen25.Qwen25_32b import Qwen25_32b
from models.qwen25.Qwen25_3b import Qwen25_3b
from models.qwen25.Qwen25_72b import Qwen25_72b
from models.qwen25.Qwen25_7b import Qwen25_7b


class ModelFactory:
    @staticmethod
    def create(model_config: ModelConfig) -> AbstractModel:
        identifier: ModelIdentifier = model_config.identifier
        # LLAMA-family
        if identifier == ModelIdentifier.LLAMA2_7b:
            model = Llama2_7b(model_config.precision, model_config.hugging_face_token,
                              model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.LAAMA2_13B:
            model = Llama2_13b(model_config.precision, model_config.hugging_face_token,
                               model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.LLAMA2_70B:
            model = Llama2_70b(model_config.precision, model_config.hugging_face_token,
                               model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.LLAMA3_1_8b:
            model = Llama31_8b(model_config.precision, model_config.hugging_face_token,
                               model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.LLAMA3_3_70b:
            model = Llama33_70b(model_config.precision, model_config.hugging_face_token,
                                model_config.store_model_disk_path)

        # QWEN-family
        elif identifier == ModelIdentifier.QWEN2_3B:
            model = Qwen25_3b(model_config.precision, model_config.hugging_face_token,
                              model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.QWEN2_7B:
            model = Qwen25_7b(model_config.precision, model_config.hugging_face_token,
                              model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.QWEN2_14B:
            model = Qwen25_14b(model_config.precision, model_config.hugging_face_token,
                               model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.QWEN2_32B:
            model = Qwen25_32b(model_config.precision, model_config.hugging_face_token,
                               model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.QWEN2_72B:
            model = Qwen25_72b(model_config.precision, model_config.hugging_face_token,
                               model_config.store_model_disk_path)

        # GEMMA-family
        elif identifier == ModelIdentifier.GEMMA2_2B:
            model = Gemma2_2b(model_config.precision, model_config.hugging_face_token,
                              model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.GEMMA2_9B:
            model = Gemma2_9b(model_config.precision, model_config.hugging_face_token,
                              model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.GEMMA2_27B:
            model = Gemma2_27b(model_config.precision, model_config.hugging_face_token,
                               model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.Gemma3_1b:
            model = Gemma3_1b(model_config.precision, model_config.hugging_face_token,
                              model_config.store_model_disk_path)

        # Other singletons
        elif identifier == ModelIdentifier.MISTRAL_7b:
            model = Mistral_7b(model_config.precision, model_config.hugging_face_token,
                               model_config.store_model_disk_path)
        elif identifier == ModelIdentifier.DEEPSEEK_7b:
            model = Deepseek_7b(model_config.precision, model_config.hugging_face_token,
                                model_config.store_model_disk_path)

        # Unknown ID
        else:
            raise ValueError(f"Unknown model identifier: {identifier}")

        return model
