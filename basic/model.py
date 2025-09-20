from transformers import BertModel

# the checkpoint name corresponds to a specific model architecture and weights. 
# In this case a BERT model with a basic architecture (12 layers, 768 hidden size, 12 attention heads) 
# and cased inputs (meaning that the uppercase/lowercase distinction is important).
model = BertModel.from_pretrained("bert-base-cased")

# the model saves the model’s weights and architecture configuration
model.save_pretrained("./directory_to_save")


# encoding text: padding inputs, truncating inputs, adding special tokens, in order to make the inputs compatible with the model