from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple
from transformers import RobertaTokenizer
import random
from copy import deepcopy

class ReWordDataLoader:
    def __init__(self, pretrained_ck: str, max_length: int):
        dataset = load_dataset('csv', data_files=["data/format_data.csv"])
        dataset = dataset['train'].train_test_split(test_size=0.05, seed=42)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_ck, add_prefix_space=True)
        self.tokenizer.add_tokens(['<ma>', '<madv>', '<mn>', '<mp>', '<ms>', '<mv>'])
        self.max_length = max_length
        random.seed(42)
        self.dataset = dataset.map(
            self.__tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

    def __tokenize_and_align_labels(self, examples):
        inputs = [inp.split(" ") for inp in examples['Input']]
        outputs = [output.split(" ") for output in examples['Output']]
        tokenized_inputs = self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
        )
        tokenized_outputs = self.tokenizer(
            outputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
        )
        tokenized_inputs['labels'] = tokenized_outputs['input_ids']
        return tokenized_inputs

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoded_inputs.items()}
        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "test"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=24)
            )
        return res
