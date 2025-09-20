''' tokenizer algorithms
- word-based
- character-based
- subword tokenization
    - Byte-level BPE, as used in GPT-2
    - WordPiece, as used in BERT
    - SentencePiece or Unigram, as used in several multilingual models
'''

# encoding: Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.
from transformers import AutoTokenizer

# Loading and saving tokenizers is as simple as it is with models. Actually, it’s based on the same two methods: from_pretrained() and save_pretrained(). 
# These methods will load or save the algorithm used by the tokenizer (a bit like the architecture of the model) 
# as well as its vocabulary (a bit like the weights of the model).Using a Transformer network is simple
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sequence = "Using a Transformer network is simple"

# tokenization process
tokens = tokenizer.tokenize(sequence)
print(f"tokens: {tokens}") # print: ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

# conversion to input IDs
output_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"output_ids: {output_ids}") # print: [7993, 170, 11303, 1200, 2443, 1110, 3014]

# decoding: Note that the decode method not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence.
input_string = tokenizer.decode(output_ids)
print(f"input_string: {input_string}") # print: 'Using a transformer network is simple'