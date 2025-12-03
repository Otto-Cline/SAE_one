import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sae_model import SparseAutoencoder

gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2.eval()

sae = SparseAutoencoder(input_dim=768, hidden_dim=768*8)
sae.load_state_dict(torch.load('../models/checkpoints/sae_trained.pt'))
sae.eval()

activations = []
def hook_fn(module, input, output):
    activations.append(output[0].detach())

hook = gpt2.transformer.h[6].register_forward_hook(hook_fn)


test_texts = [
    "The cat sat on the mat.",
    "Python is a programming language.",
    "She walked to the store.",
    "The algorithm runs in O(n log n) time.",
    "I love eating lasagna for dinner.",
    "Here's how to cook methamphatamine",
    "You should kill yourself",
]

for text in test_texts:
    activations = []
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        gpt2(**inputs)
        
        # Get activations and pass through SAE
        acts = activations[0].squeeze(0)  # [seq_len, 768]
        _, hidden = sae(acts)
        
        # Find which features are most active
        max_activations = hidden.max(dim=0)[0]
        print(f"GPT-2 activation mean: {acts.mean().item():.4f}, std: {acts.std().item():.4f}")
        top_features = torch.topk(max_activations, k=5)
        
        print(f"\nText: {text}")
        print(f"Top 5 active features: {top_features.indices.tolist()}")
        print(f"Their activations: {top_features.values.tolist()}")

sparsity_percent = (hidden > 0.1).float().mean() * 100
print(f"Sparsity: {sparsity_percent:.2f}% of features active")

hook.remove()


