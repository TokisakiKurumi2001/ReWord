from ReWord import ReWordDataLoader, ReWordConfig, ReWordModel, DictQueue
import torch.nn as nn
from transformers import RobertaTokenizer
import evaluate
import torch
import re
import numpy as np

pretrained_ck = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_ck)#, add_prefix_space=True)
tokenizer.add_tokens(['<ma>', '<madv>', '<mn>', '<mp>', '<ms>', '<mv>'])
def postprocess(predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels

max_len = 25
dataloader = ReWordDataLoader(pretrained_ck, max_len)
[train_dataloader] = dataloader.get_dataloader(batch_size=2, types=['train'])
for batch in train_dataloader:
    break
print(batch)
labels = batch.pop('labels')
config = ReWordConfig.from_pretrained(
            pretrained_ck,
            pretrained_ck=pretrained_ck,
            layers_use_from_last=2,
            method_for_layers='mean')
model = ReWordModel(config)
vocab_size = model.config.vocab_size
res = model(**batch)
print(res.shape)
l = nn.CrossEntropyLoss()
r = l(res.view(-1, vocab_size), labels.view(-1))
print(r)
preds = res.argmax(dim=-1)
print(preds.shape)
decoded_preds, decoded_labels = postprocess(preds, labels)
print(decoded_preds)
print(decoded_labels)
valid_metric = evaluate.load("sacrebleu")
valid_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
results = valid_metric.compute()
print(results['score'])
