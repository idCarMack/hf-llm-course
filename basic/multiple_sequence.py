# handling multiple sequences: models expect a batch of inputs
# strategies:
# - padding: add padding tokens to the sequences to make them all the same length
# - attention mask: mask the padding tokens to avoid them from affecting the model's attention mechanism
# - truncation: truncate the sequences to avoid exceeding the model's maximum sequence length

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"ids: {ids}")
input_ids = torch.tensor([ids]) # directly pass the ids is invalid, we need to add a batch dimension
print(f"input_ids: {input_ids}")

# This line will fail.
outputs = model(input_ids)
print(f"outputs: {outputs}")

# without attention mask: the model outputs different results for the same input
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

# with attention mask: the model outputs the same result for the same input
outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)