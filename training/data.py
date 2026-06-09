# training/data.py
from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataloader(config):
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",
                           name="sample-10BT",
                           split="train",
                           streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    # Or train your own BPE tokenizer on the dataset for cleaner vocab

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=config.max_seq_len)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids'])

# For streaming, you cannot use map easily. Alternative:
def gen():
    for example in dataset:
        tokens = tokenizer(example['text'], truncation=True, max_length=config.max_seq_len)
        yield {'input_ids': tokens['input_ids']}

from torch.utils.data import IterableDataset, DataLoader
class StreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __iter__(self):
        for example in self.dataset:
            tokens = self.tokenizer(example['text'], truncation=True, max_length=self.max_length)
            yield {'input_ids': tokens['input_ids']}