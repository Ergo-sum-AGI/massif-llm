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
    ...