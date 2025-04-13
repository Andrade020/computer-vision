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
# Configurações utilizadas no treinamento
LATENT_DIM = 128
EMBEDDING_DIM = 16
# Se você salvou com a mesma estrutura do ImageFolder, pode utilizar o dataset para mapear o rótulo da classe.
# Por exemplo, se o dataset tinha o seguinte mapeamento:
#   dataset.classes = ['a', 'b', ..., '_A', '_B', ...]
# Você pode ter o seguinte:
DATA_DIR = r"C:\Users\Leo\Desktop\Portfolio\computer-vision\handwritten_text\segmented\recognized"

# Se você ainda não tiver o dataset carregado, pode carregar novamente para obter o mapeamento:
from torchvision import datasets
import os

def is_valid_file(path):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
    return any(path.lower().endswith(ext) for ext in IMG_EXTENSIONS)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform, is_valid_file=is_valid_file)
num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# Criação do modelo e carregamento dos pesos treinados
model = ConditionalVAE(latent_dim=LATENT_DIM, embedding_dim=EMBEDDING_DIM, num_classes=num_classes)
model.load_state_dict(torch.load("cvae_handwritten_letters.pth", map_location=torch.device("cpu")))
model.eval()  # Coloca o modelo em modo de avaliação

# Função para gerar e exibir uma letra específica
def gerar_letra(letra, modelo, dataset, latent_dim=LATENT_DIM):
    # Mapeia a letra para o índice do dataset.
    # Observe: se as letras maiúsculas estão com prefixo "_" no diretório, ajuste conforme necessário.
    if letra.isupper():
        chave = f"_{letra}"  # Exemplo: para "A", a chave será "_A"
    else:
        chave = letra.lower()
    
    if chave not in dataset.class_to_idx:
        print(f"A letra '{letra}' não foi encontrada no dataset.")
        return

    label_index = dataset.class_to_idx[chave]
    label = torch.tensor([label_index])
    
    # Gera um vetor de ruído a partir de uma distribuição normal
    z = torch.randn(1, latent_dim)
    
    # Decodifica para gerar a imagem da letra
    with torch.no_grad():
        imagem_gerada = modelo.decode(z, label)
    
    # Converte para numpy e exibe
    imagem_np = imagem_gerada.cpu().squeeze().numpy()
    
    plt.figure(figsize=(3,3))
    plt.imshow(imagem_np, cmap="gray")
    plt.title(f"Letra gerada: {letra}")
    plt.axis("off")
    plt.show()


# Exemplo: gerar a letra "A"
gerar_letra("x", model, dataset)
