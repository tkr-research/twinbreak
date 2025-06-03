import os
from typing import List, Tuple

import torch

import json
from models.AbstractModel import AbstractModel


class DatasetBucket:
    """
    This class can hold data from our JSON file datasets
    """

    def __init__(self, dataset_identifier: str):
        self.__identifier: str = dataset_identifier
        self.__harmless_prompts: List[str] = []
        self.__harmful_prompts: List[str] = []
        self.__harmless_prompts_tokenized: torch.Tensor | None = None
        self.__harmless_mask_tokenized: torch.Tensor | None = None
        self.__harmful_prompts_tokenized: torch.Tensor | None = None
        self.__harmful_mask_tokenized: torch.Tensor | None = None
        self.__is_tokenized: bool = False
        self.__categories: List[str] = []
        self.__input_token_length: int = 0

        dataset_folder_name = os.path.dirname(__file__)
        with open(f'{dataset_folder_name}/json/{self.__identifier}', 'r') as f:
            ds = json.load(f)
            for item in ds:
                self.__harmless_prompts.append(item['benign'])
                self.__harmful_prompts.append(item['mal'])
                self.__categories.append(item['category'])

    def get_size(self) -> int:
        return len(self.__harmful_prompts)

    def get_sample(self, idx: int) -> Tuple[str, str, str]:
        return self.__harmless_prompts[idx], self.__harmful_prompts[idx], self.__categories[idx]

    def tokenize(self, model: AbstractModel) -> None:
        all_prompts = []
        all_prompts.extend(self.__harmless_prompts)
        all_prompts.extend(self.__harmful_prompts)
        all_prompts = [model.get_chat_template().format(instruction=prompt) for prompt in all_prompts]
        all_prompts_tokenized = model.tokenizer(all_prompts, truncation=False, padding=True, return_tensors='pt')

        self.__harmless_prompts_tokenized = all_prompts_tokenized.input_ids[0:self.get_size()]
        self.__harmless_mask_tokenized = all_prompts_tokenized.attention_mask[0:self.get_size()]
        self.__harmful_prompts_tokenized = all_prompts_tokenized.input_ids[self.get_size():]
        self.__harmful_mask_tokenized = all_prompts_tokenized.attention_mask[self.get_size():]

        self.__input_token_length = self.__harmful_prompts_tokenized.shape[1]

        self.__is_tokenized = True

    def get_harmless_prompts_tokenized(self) -> torch.Tensor:
        if not self.__is_tokenized:
            raise Exception('DatasetBucket is not tokenized.')
        return self.__harmless_prompts_tokenized

    def get_harmless_mask_tokenized(self) -> torch.Tensor:
        if not self.__is_tokenized:
            raise Exception('DatasetBucket is not tokenized.')
        return self.__harmless_mask_tokenized

    def get_harmful_prompts_tokenized(self) -> torch.Tensor:
        if not self.__is_tokenized:
            raise Exception('DatasetBucket is not tokenized.')
        return self.__harmful_prompts_tokenized

    def get_harmful_mask_tokenized(self) -> torch.Tensor:
        if not self.__is_tokenized:
            raise Exception('DatasetBucket is not tokenized.')
        return self.__harmful_mask_tokenized

    def get_input_token_length(self) -> int:
        if not self.__is_tokenized:
            raise Exception('DatasetBucket is not tokenized.')
        return self.__input_token_length
