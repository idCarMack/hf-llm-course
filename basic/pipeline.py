from transformers import pipeline
import torch


# 0. Sentiment Analysis pipeline
    # - preprocessing with tokenizer
    # - passing the inputs through the model: model inference
    # - postprocessing: softmax to get the probabilities etc
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

classifier = pipeline("sentiment-analysis", device=device)

result = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

print(f"\n\nSentiment Analysis pipeline...............................:\n {result}")


# 1. Tokenizer
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(f"\n\nTokenizer...............................:\n {inputs}")    # print: [{'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]


# 2. Model inference
# 2.1. AutoModel
from transformers import AutoModel
# We can download our pretrained model the same way we did with our tokenizer. 
# 🤗 Transformers provides an AutoModel class which also has a from_pretrained() method
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
# print: BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
# BaseModelOutput(last_hidden_state=tensor([[[-0.1798,  0.2333,  0.6321,  ..., -0.3017,  0.5008,  0.1481],
#          [ 0.2758,  0.6497,  0.3200,  ..., -0.0760,  0.5136,  0.1329],
#          [ 0.9046,  0.0985,  0.2950,  ...,  0.3352, -0.1407, -0.6464],
#          ...,
#          [ 0.1466,  0.5661,  0.3235,  ..., -0.3376,  0.5100, -0.0561],
#          [ 0.7500,  0.0487,  0.1738,  ...,  0.4684,  0.0030, -0.6084],
#          [ 0.0519,  0.3729,  0.5223,  ...,  0.3584,  0.6500, -0.3883]],

#         [[-0.2937,  0.7283, -0.1497,  ..., -0.1187, -1.0227, -0.0422],
#          [-0.2206,  0.9384, -0.0951,  ..., -0.3643, -0.6605,  0.2407],
#          [-0.1536,  0.8988, -0.0728,  ..., -0.2189, -0.8528,  0.0710],
#          ...,
#          [-0.3017,  0.9002, -0.0200,  ..., -0.1082, -0.8412, -0.0861],
#          [-0.3338,  0.9674, -0.0729,  ..., -0.1952, -0.8181, -0.0634],
#          [-0.3454,  0.8824, -0.0426,  ..., -0.0993, -0.8329, -0.1065]]],
#        grad_fn=<NativeLayerNormBackward0>), hidden_states=None, attentions=None)
print(f"\n\nAutoModel outputs...............................:\n {outputs}") 

# The vector output by the Transformer module is usually large. It generally has three dimensions:
# - Batch size: The number of sequences processed at a time (2 in our example).
# - Sequence length: The length of the numerical representation of the sequence (16 in our example).
# - Hidden size: The vector dimension of each model input.
print(f"\n\nAutoModel outputs.last_hidden_state.shape...............................:\n {outputs.last_hidden_state.shape}")   # print: torch.Size([2, 16, 768]) 

# 2.2. AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(f"\n\nAutoModelForSequenceClassification outputs...............................:\n {outputs}")
print(f"\n\nAutoModelForSequenceClassification outputs.logits...............................:\n {outputs.logits}")
print(f"\n\nAutoModelForSequenceClassification outputs.logits.shape...............................:\n {outputs.logits.shape}")   # print: torch.Size([2, 2])

# 3. Postprocessing 
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(f"\n\nPostprocessing...............................:\n {model.config.id2label}")
print(f"\n\nPostprocessing...............................:\n {predictions}")

