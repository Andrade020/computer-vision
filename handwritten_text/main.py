import cv2
import os
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from glob import glob

# Defina os diretórios de entrada e saída
input_dir = r"handwritten_text\original"  # Pasta com imagens JPG
output_dir = r"handwritten_text\segmented"

# Cria a estrutura de pastas de saída, se não existir
recognized_dir = os.path.join(output_dir, "recognized")
not_recognized_dir = os.path.join(output_dir, "nao_reconhecidos")
words_dir = os.path.join(output_dir, "palavras_agrupadas")

os.makedirs(recognized_dir, exist_ok=True)
os.makedirs(not_recognized_dir, exist_ok=True)
os.makedirs(words_dir, exist_ok=True)

# Lista todos os arquivos JPG na pasta de origem
image_paths = glob(os.path.join(input_dir, "*.jpg"))

# Função para salvar a imagem em determinado caminho
def save_roi(roi, dest_dir, base_name, index, prefix=""):
    filename = f"{base_name}_{prefix}{index}.jpg"
    cv2.imwrite(os.path.join(dest_dir, filename), roi)

# Processa cada imagem
for image_path in image_paths:
    # Nome base da imagem (sem extensão)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Carrega e converte para escala de cinza
    image = cv2.imread(image_path)
    if image is None:
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplica threshold para binarização.
    # Utilizamos THRESH_BINARY_INV para que os caracteres fiquem em branco (foreground)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Encontra os contornos externos, que poderão representar caracteres ou grupos de caracteres
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtra pequenas regiões que podem ser ruído
        if w < 10 or h < 10:
            continue
        
        # Extrai a região de interesse (ROI) do caractere ou grupo de caracteres
        roi = thresh[y:y+h, x:x+w]
        
        # Se a largura for significativamente maior que a altura, pode ser que haja agrupamento de letras (palavra "grudada")
        if w > 1.5 * h:
            save_roi(roi, words_dir, base_name, i, prefix="word_")
            continue
        
        # Configuração para reconhecer um único caractere (PSM 10: trata a imagem como um único caractere)
        # Aqui restringimos a lista de caracteres a letras (maiúsculas e minúsculas)
        config = '--psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        recognized_text = pytesseract.image_to_string(roi, config=config).strip()
        
        # Verifica se o OCR retornou um único caractere
        if len(recognized_text) != 1:
            # Caso não esteja claro ou tenha retornado mais de um caractere,
            # salve na pasta "nao_reconhecidos"
            save_roi(roi, not_recognized_dir, base_name, i, prefix="nr_")
        else:
            # Caso o OCR identifique um caractere, cria (se ainda não existir) a pasta
            # referente a ele e salva a imagem
            letter_dir = os.path.join(recognized_dir, recognized_text)
            os.makedirs(letter_dir, exist_ok=True)
            save_roi(roi, letter_dir, base_name, i)
