import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sae_model import SparseAutoencoder

#load activations
activations = torch.load('../data/activations/layer6_activations.pt')
print(f"Loaded {activations.shape[0]} activation vectors")

#create dataset and dataloader
dataset = TensorDataset(activations)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

#initialize model
model = SparseAutoencoder(input_dim=768, hidden_dim=768*8, sparsity_coef=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    total_recon = 0
    total_sparsity = 0
    
    for batch in dataloader:
        x = batch[0]
        
        #forward pass
        reconstructed, hidden = model(x)
        loss, recon_loss, sparsity_loss = model.loss(x, reconstructed, hidden)
        
        #backward pass, calculates gradients (how much each weight contributed
        #to the error) and updates weights to reduce loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #track losses
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_sparsity += sparsity_loss.item()
    
    #some epoch stats
    n_batches = len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Total Loss: {total_loss/n_batches:.4f}")
    #how good our reconstructions are
    print(f"  Recon Loss: {total_recon/n_batches:.4f}")
    #how few neurons we used
    print(f"  Sparsity Loss: {total_sparsity/n_batches:.4f}")

#save trained model
torch.save(model.state_dict(), '../models/checkpoints/sae_trained.pt')
print("Model saved, BANG")