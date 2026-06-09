# training/tinystories.py
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_tinystories_dataloader(tokenizer, batch_size=8, max_length=256):
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)