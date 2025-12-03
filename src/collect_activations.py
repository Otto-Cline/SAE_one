import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

#contains the pointers to the tensors for each forward pass (1000 in this script)
all_activations = []


#hook - gets all activations in layer, append them to all_activations
def hook_fn(module, input, output):
    acts = output[0].detach()
    acts = acts.view(-1, 768)
    all_activations.append(acts)

#defines which layer to hook and calls hook on that layer
layer_6 = model.transformer.h[6]
hook = layer_6.register_forward_hook(hook_fn)

#download some shakespear to use as text
url = "https://www.gutenberg.org/files/100/100-0.txt"
response = requests.get(url)
full_text = response.text

#split shakespear text into chunks bla bla
chunks = []
words = full_text.split()
chunk_size = 100
for i in range(0, len(words), chunk_size):
    chunk = ' '.join(words[i:i+chunk_size])
    chunks.append(chunk)
    if len(chunks) >= 1000:
        break

#for each example in the dataset, turn text into tokens(numbers).
#then do a forward pass through the model with these tokens
for i, text in enumerate(chunks):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    
    with torch.no_grad():
        model(**inputs)
    
    if i % 100 == 0:
        print(f"Processed {i} examples...")


hook.remove()

#stacks all the tensors from each forward pass on top of each other
# to form one big tensor for training the SAE
activations_tensor = torch.cat(all_activations, dim=0)
print(f"Total activations collected: {activations_tensor.shape}")

torch.save(activations_tensor, '../data/activations/layer6_activations.pt')