from random import sample
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")

# print the dataset
'''
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
'''
print(raw_datasets)

# print the first example
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

from transformers import AutoTokenizer

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# The Hugging Face dataset stores columns as an Arrow "Column" object, which the
# tokenizer does not recognise as a plain list. Convert to a list before passing.
print(type(raw_datasets["train"]["sentence1"]))
#train_sentence1 = list(raw_datasets["train"]["sentence1"])
#train_sentence2 = list(raw_datasets["train"]["sentence2"])

#tokenized_sentences_1 = tokenizer(train_sentence1)
#tokenized_sentences_2 = tokenizer(train_sentence2)

tokenized_dataset = tokenizer(list(raw_datasets["train"]["sentence1"]), list(raw_datasets["train"]["sentence2"]), padding=True, truncation=True)
print(len(tokenized_dataset))



def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]  # [50, 59, 47, 67, 59, 50, 62, 32]

batch = data_collator(samples)
'''
{
    'attention_mask': torch.Size([8, 67]),
    'input_ids': torch.Size([8, 67]),
    'token_type_ids': torch.Size([8, 67]),
    'labels': torch.Size([8])
}
'''
print({k: v.shape for k, v in batch.items()})


