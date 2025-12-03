import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

#contains the pointers to the tensors for each token
activations = []

#1. make a hook function that gets called during forward pass
def hook_fn(module, input, output):
    #output[0] is an activation tensor
    #shape: [batch_size, sequence length, hidden_dim]
    activations.append(output[0].detach())

layer_6 = model.transformer.h[6]

# this line says: when data passes through layer 6, 
# also run this extra function which grabs the outputs.
# the model, what went into the layer, and what came out of the layer are
# automatically called with hook_fn
hook = layer_6.register_forward_hook(hook_fn)


text = "The cat sat on the mat and looked around"
inputs = tokenizer(text, return_tensors='pt')

# torch automatically tracks operations to compute gradients for training,
# but we're just extracting data here, so no_grad tells it "don't bother"
with torch.no_grad():
    outputs = model(**inputs)

# needed so future forward passes don't keep appending to activations
hook.remove()

print(f"Captured activation shape: {activations[0].shape}")
print(f"This is [batch=1, tokens={activations[0].shape[1]}, hidden_dim={activations[0].shape[2]}]")