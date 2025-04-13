import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
import matplotlib.pyplot as plt

# Configurações gerais
DATA_DIR = r"C:\Users\Leo\Desktop\Portfolio\computer-vision\handwritten_text\segmented\recognized"
BATCH_SIZE = 16
IMAGE_SIZE = 64  # redimensiona para 64x64
LATENT_DIM = 128
EMBEDDING_DIM = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation e pré-processamento
# Como os dados são poucos, usamos transformações que incluem rotações e pequenas deformações.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # garante que a imagem seja 1 canal
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),  # converte para tensor com valores em [0,1]
])

# Utiliza ImageFolder: cada subpasta corresponde a uma classe.
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

num_classes = len(dataset.classes)
print(f"Número de classes: {num_classes}")
print("Classes:", dataset.classes)

# Modelo: Conditional VAE
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim, embedding_dim, num_classes):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # Embedding dos rótulos
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Encoder: de imagem (1x64x64) para um vetor (flatten de tamanho 256*4*4 = 4096)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 8)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder: recebe z concatenado com o embedding da classe
        self.fc_decode = nn.Linear(latent_dim + embedding_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (64, 16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (32, 32, 32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # (1, 64, 64)
            nn.Sigmoid()  # saída em [0,1]
        )

    def encode(self, x):
        batch_size = x.size(0)
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(batch_size, -1)  # flatten
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        # Obtém o embedding do rótulo e concatena com z
        label_emb = self.label_embedding(labels)
        z_cond = torch.cat([z, label_emb], dim=1)
        x_decoded = self.fc_decode(z_cond)
        x_decoded = x_decoded.view(-1, 256, 4, 4)
        x_recon = self.decoder(x_decoded)
        return x_recon

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar

# Inicialização do modelo, otimizador e função de perda
model = ConditionalVAE(latent_dim=LATENT_DIM, embedding_dim=EMBEDDING_DIM, num_classes=num_classes).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
bce_loss = nn.BCELoss(reduction='sum')

def loss_function(recon_x, x, mu, logvar):
    # Perda de reconstrução + divergência KL
    recon_loss = bce_loss(recon_x, x)
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Função para treinar o modelo
def train(model, data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(data_loader)}] Loss: {loss.item()/len(data):.4f}")
    print(f"==> Epoch [{epoch}] Average loss: {train_loss / len(data_loader.dataset):.4f}")

# Função para salvar amostras geradas (para visualizar a geração condicional)
def generate_and_save(model, epoch, sample_count=16):
    model.eval()
    # Gera rótulos aleatórios ou, por exemplo, de 0 até num_classes-1
    labels = torch.arange(0, num_classes, dtype=torch.long).to(DEVICE)
    labels = labels.repeat(int(np.ceil(sample_count / num_classes)))[:sample_count]
    with torch.no_grad():
        z = torch.randn(sample_count, LATENT_DIM).to(DEVICE)
        samples = model.decode(z, labels).cpu()
    # Salva a imagem em grade
    grid = utils.make_grid(samples, nrow=4)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.title(f"Amostras geradas - Epoch {epoch}")
    plt.axis("off")
    plt.savefig(f"epochs/generated_epoch_{epoch}.png")
    plt.close()

# Laço de treinamento
for epoch in range(1, NUM_EPOCHS + 1):
    train(model, data_loader, optimizer, epoch)
    generate_and_save(model, epoch)

# Salvando o modelo treinado
torch.save(model.state_dict(), "cvae_handwritten_letters.pth")
print("Modelo salvo como 'cvae_handwritten_letters.pth'")
