import abc
from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.experiments.dtype.DTypeConfigFactory import DTypeConfigFactory
from configs.experiments.dtype.PrecisionConfig import PrecisionConfig


class AbstractModel(metaclass=abc.ABCMeta):
    def __init__(self, precision: PrecisionConfig, hugging_face_token: str, store_model_disk_path: str):
        if precision == PrecisionConfig.BIT8:
            self.model = AutoModelForCausalLM.from_pretrained(self.get_huggingface_path(),
                                                              token=hugging_face_token,
                                                              load_in_8bit=True,
                                                              trust_remote_code=True,
                                                              device_map='auto',
                                                              cache_dir="/data/hamid/huggingface")
        else:
            dtype = DTypeConfigFactory().create(precision)
            self.model = AutoModelForCausalLM.from_pretrained(self.get_huggingface_path(),
                                                              torch_dtype=dtype,
                                                              token=hugging_face_token,
                                                              trust_remote_code=True,
                                                              device_map='auto',
                                                              cache_dir=store_model_disk_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.get_huggingface_path(),
                                                       token=hugging_face_token,
                                                       trust_remote_code=True,
                                                       use_fast=False,
                                                       cache_dir=store_model_disk_path)

        self.initialize_tokenizer()
        self.initialize_model()
        self.model.eval()
        self.model.requires_grad_(False)

    @abc.abstractmethod
    def initialize_tokenizer(self) -> None:
        """Make initial adjustments to the tokenizer"""

    def initialize_model(self) -> None:
        """Make initial adjustments to the model"""
        pass

    @abc.abstractmethod
    def get_chat_template(self) -> str:
        """Provide the chat template for the model"""

    def get_layer_list(self):
        """Get the module containing the layers from the model"""
        return self.model.model.layers

    def get_model_head(self):
        """Get the module containing the layers from the model"""
        return self.model.lm_head

    def get_hidden_layer_num(self) -> int:
        """Get the number of the hidden layers"""
        return self.model.model.config.num_hidden_layers

    @abc.abstractmethod
    def get_huggingface_path(self) -> str:
        """Get the path for HiggingFace"""

    @abc.abstractmethod
    def extract_prompt_and_response(self, output: str) -> Tuple[str, str]:
        """Extract the prompt and response from the model output."""
