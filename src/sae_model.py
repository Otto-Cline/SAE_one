import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768*8, sparsity_coef=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef

        #just like training a normal net, just defined numbers in above lines
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    
    def forward(self, x):
        #encode with relu
        hidden = torch.relu(self.encoder(x))
        
        #decode back to original dimension
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden

    def loss(self, x, reconstructed, hidden):
        #basic MSE, penalize model for getting the answer wrong
        recon_loss = torch.mean((x - reconstructed) ** 2)
        
        #penalize the model for activating too many neurons in hidden layer
        sparsity_loss = torch.mean(torch.abs(hidden))
        
        #total loss
        #by using both the regular loss function and the sparsity loss,
        #we say: "reconstruct the input well, but also try to use as few
        #hidden neurons as possible"
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss
        
        return total_loss, recon_loss, sparsity_loss