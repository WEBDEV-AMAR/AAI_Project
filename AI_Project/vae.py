import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import VAE

def loss_fn(recon, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon, x.view(-1,784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        total = 0
        for x,_ in loader:
            x = x.to(device)

            recon, mu, logvar = model(x)
            loss = loss_fn(recon, x, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total}")

    torch.save(model.state_dict(), "vae_model.pth")
    print("VAE saved!")