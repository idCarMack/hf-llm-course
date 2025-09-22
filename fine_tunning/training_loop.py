from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

''' 
    the following is what the Trainer does for us
'''

# S1: prepare for training
# S1.1: Postprocess tokenized dataset
#   - Remove the columns corresponding to values the model does not expect (like the sentence1 and sentence2 columns).
#   - Rename the column label to labels (because the model expects the argument to be named labels).
#   - Set the format of the datasets so they return PyTorch tensors instead of lists.
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
# remaining only the columns the model expects
tokenized_datasets["train"].column_names    # ['input_ids', 'token_type_ids', 'attention_mask', 'labels'] 

# S1.2: Build dataloaders & Load model
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
''' 
{'attention_mask': torch.Size([8, 65]),
 'input_ids': torch.Size([8, 65]),
 'labels': torch.Size([8]),
 'token_type_ids': torch.Size([8, 65])}
'''
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}


from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
'''
tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
'''
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)


# S1.3: optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# S1.4: learning rate scheduler
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# S1.5: device setup
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# S2: training loop
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()} # Move every tensor to the target device
        outputs = model(**batch)    # Forward pass — run the model to get outputs
        loss = outputs.loss         
        loss.backward()             # Backward pass — compute gradients of the loss w.r.t. model parameters

        optimizer.step()            # Optimizer step — update parameters using accumulated gradients
        lr_scheduler.step()         # Scheduler step — update the learning rate according to the schedule
        optimizer.zero_grad()       # clear grads so the next batch starts fresh (PyTorch accumulates by default)
        progress_bar.update(1)

# S3: evaluation loop
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())