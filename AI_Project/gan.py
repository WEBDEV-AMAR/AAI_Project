import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1,1,28,28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(20):
        for real,_ in loader:
            real = real.to(device)
            bs = real.size(0)

            real_labels = torch.ones(bs,1).to(device)
            fake_labels = torch.zeros(bs,1).to(device)

            # Train D
            z = torch.randn(bs,100).to(device)
            fake = G(z)

            loss_D = criterion(D(real), real_labels) + criterion(D(fake.detach()), fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train G
            z = torch.randn(bs,100).to(device)
            fake = G(z)

            loss_G = criterion(D(fake), real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch+1}")

    torch.save(G.state_dict(), "gan_model.pth")
    print("GAN saved!")