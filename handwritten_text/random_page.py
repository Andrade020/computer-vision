import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms, utils

# ================================
# 1. Definição do Modelo Conditional VAE
# ================================
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim, embedding_dim, num_classes):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # Embedding dos rótulos
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Encoder: de imagem (1x64x64) para um vetor
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # Saída: 32x32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),    # Saída: 64x16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # Saída: 128x8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Saída: 256x4x4
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder: recebe (z + embedding do rótulo) e reconstrói a imagem
        self.fc_decode = nn.Linear(latent_dim + embedding_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Saída: 128x8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Saída: 64x16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Saída: 32x32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # Saída: 1x64x64
            nn.Sigmoid()  # Garante saída em [0, 1]
        )

    def encode(self, x):
        batch_size = x.size(0)
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.view(batch_size, -1)  # Flatten
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        label_emb = self.label_embedding(labels)
        z_cond = torch.cat([z, label_emb], dim=1)
        x_decoded = self.fc_decode(z_cond)
        x_decoded = x_decoded.view(-1, 256, 4, 4)
        x_recon = self.decoder(x_decoded)
        return x_recon

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

# ================================
# 2. Variáveis e Dataset
# ================================
# Caminho para a pasta com as letras (cada subpasta contém imagens de um caractere)
DATA_DIR = r"C:\Users\Leo\Desktop\Portfolio\computer-vision\handwritten_text\segmented\recognized"

# Função para garantir que arquivos com extensões em maiúsculo também sejam aceitos
def is_valid_file(path):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
    return any(path.lower().endswith(ext) for ext in IMG_EXTENSIONS)

# Transformações para as imagens (converter para escala de cinza e redimensionar para 64x64)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Carrega o dataset usando ImageFolder
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform, is_valid_file=is_valid_file)
num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# Parâmetros do modelo
LATENT_DIM = 128
EMBEDDING_DIM = 16

# ================================
# 3. Carregar o Modelo Treinado
# ================================
model = ConditionalVAE(latent_dim=LATENT_DIM, embedding_dim=EMBEDDING_DIM, num_classes=num_classes)
model_path = "cvae_handwritten_letters.pth"  # Certifique-se de que este arquivo existe
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print("Modelo carregado com sucesso!")
else:
    print(f"Atenção: O modelo '{model_path}' não foi encontrado.")
    exit(1)
model.eval()  # Modo de avaliação

# ================================
# 4. Função para Gerar Letra Aleatória (mantendo as cores originais)
# ================================
def generate_random_letter(model, dataset, latent_dim):
    # Seleciona aleatoriamente uma letra entre as classes disponíveis
    letter = random.choice(dataset.classes)
    label_index = dataset.class_to_idx[letter]
    label = torch.tensor([label_index])
    # Gera um vetor de ruído
    z = torch.randn(1, latent_dim)
    with torch.no_grad():
        output = model.decode(z, label).cpu()
    # Converte o tensor para uma imagem em escala de cinza (64x64)
    img = output.squeeze().numpy()
    img = (img * 255).astype('uint8')
    # NÃO inverte as cores; a imagem permanece com o fundo branco e traços escuros conforme gerada
    return img, letter

# ================================
# 5. Função para Gerar a "Folha" de Texto com fundo preto e letras menores
# ================================
def generate_text_page(model, dataset, latent_dim, page_width=800, page_height=1000,
                       letter_spacing=5, line_spacing=10, margin_x=20, margin_y=20,
                       letter_scale=0.5):
    # Cria um canvas com fundo preto
    page = np.zeros((page_height, page_width), dtype='uint8')
    current_x = margin_x
    current_y = margin_y

    # Gera uma letra de exemplo para definir dimensões após redimensionamento
    sample_letter, _ = generate_random_letter(model, dataset, latent_dim)
    sample_letter = cv2.resize(sample_letter, (0, 0), fx=letter_scale, fy=letter_scale, interpolation=cv2.INTER_AREA)
    letter_height, letter_width = sample_letter.shape

    # Preenche o canvas com letras enquanto houver espaço vertical
    while current_y + letter_height < page_height - margin_y:
        # Se não houver espaço na linha, pula para a próxima linha
        if current_x + letter_width > page_width - margin_x:
            current_x = margin_x
            current_y += letter_height + line_spacing
            if current_y + letter_height > page_height - margin_y:
                break

        # Gera uma letra aleatória e redimensiona-a
        letter_img, letter = generate_random_letter(model, dataset, latent_dim)
        letter_img = cv2.resize(letter_img, (0, 0), fx=letter_scale, fy=letter_scale, interpolation=cv2.INTER_AREA)
        h, w = letter_img.shape

        # Posição da letra na página (coloca somente se couber na área)
        if current_y + h <= page_height and current_x + w <= page_width:
            page[current_y:current_y+h, current_x:current_x+w] = letter_img

        current_x += w + letter_spacing

    return page

# ================================
# 6. Gerar a Página e Exibir/Salvar o Resultado
# ================================
if __name__ == '__main__':
    # Gera a página; ajuste letter_scale para definir o tamanho das letras
    page_image = generate_text_page(model, dataset, latent_dim=LATENT_DIM, page_width=800, page_height=1000,
                                    letter_scale=0.5)
    
    # Exibe a imagem usando matplotlib
    plt.figure(figsize=(8, 10))
    plt.imshow(page_image, cmap="gray")
    plt.title("Página Gerada com Letras Aleatórias (Fundo Preto)")
    plt.axis("off")
    plt.show()
    
    # Salva a imagem gerada
    output_path = "generated_text_page.png"
    cv2.imwrite(output_path, page_image)
    print(f"Página gerada salva como '{output_path}'")
