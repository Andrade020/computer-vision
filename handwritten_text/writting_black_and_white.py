# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import random
from glob import glob
import unicodedata
import sys

# ------------------------- Funções auxiliares -------------------------

def normalize_for_filesystem(letter, form="NFC"):
    # Apenas normaliza o caractere sem re-encodificação desnecessária
    return unicodedata.normalize(form, letter)

def get_folder_for_letter(letter, prefix=""):
    """
    Procura a pasta correspondente ao caractere fornecido,
    testando as formas NFC e NFD (caso você não tenha renomeado o arquivo exatamente)
    e usando o prefixo quando necessário.
    """
    for form in ("NFC", "NFD"):
        letter_norm = unicodedata.normalize(form, letter)
        folder_path = os.path.join(data_dir, f"{prefix}{letter_norm}")
        if os.path.exists(folder_path) and glob(os.path.join(folder_path, "*.jpg")):
            return folder_path, letter_norm
    return None, None

def remove_accents(char):
    """
    Remove acentos de um caractere, mas mantém o 'ç' intacto.
    """
    if char in ['ç', 'Ç']:
        return char
    nfkd = unicodedata.normalize('NFKD', char)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

# Defina o diretório onde estão as amostras
data_dir = "handwritten_text/segmented/recognized"

# Grupos para alguns ajustes de escala (não é necessário alterar)
high_letters = set("htflkbd")
descenders = set("jgqpç")

target_height = 46  # altura base para letras sem traços altos
descender_shift = int(45 * (target_height / 100.0))  # aproximadamente 7

def choose_letter_for_folder(letter):
    """
    Retorna o caractere (ou string) a ser utilizado para carregar a amostra.
    Aqui é realizado o mapeamento para os caracteres especiais:
      'é' -> 'eh'
      'ê' -> 'ee'
      'ç' -> 'cc'
    Também é considerado o caso de letras maiúsculas, adicionando o prefixo "_" se necessário.
    """
    # Dicionário de mapeamento dos caracteres especiais
    accent_mapping = {
        "é": "eh",
        "ê": "ee",
        "ç": "cc",
        "ã": "aaa",
        "á": "aa",
        "à": "aaaa", 
        "â": "aaaaa",
        "ó": "oo", 
        "õ": "ooo",
        "ô": "ooooo",
        "ú": "uu",
        "í": "ii",
        ".": ",,", 
        "?": ",22", 
        "\"": "\'\'", 
        ":" : "--"




    }
    
    # Se o caractere estiver no mapeamento, usamos o mapeamento
    if letter in accent_mapping:
        letter_mapped = accent_mapping[letter]
    else:
        letter_mapped = letter

    if letter_mapped == " ":
        return letter_mapped

    # Tenta encontrar a pasta usando o caractere (já mapeado)
    if letter_mapped.isupper():
        folder, letter_norm = get_folder_for_letter(letter_mapped, prefix="_")
        if folder is not None:
            return letter_norm
        else:
            # Se não encontrar, mantém o mapeamento sem acentuação
            return letter_mapped
    else:
        folder, letter_norm = get_folder_for_letter(letter_mapped)
        if folder is not None:
            return letter_norm
        else:
            # Se não encontrar a pasta, retorna o caractere mapeado
            return letter_mapped

def load_letter_image(letter):
    """
    Carrega uma imagem de amostra para o caractere solicitado, garantindo fundo branco (255)
    e traços em preto (0). Ajusta o fator de escala conforme se a letra for alta ou tiver descendente.
    """
    # Para letras maiúsculas, tenta obter a pasta com prefixo "_"
    if letter.isupper():
        folder, letter_norm = get_folder_for_letter(letter, prefix="_")
        if folder is not None:
            scale_factor = 1.8
        else:
            # Se não encontrar a pasta de maiúsculas, tenta com a minúscula
            folder, letter_norm = get_folder_for_letter(letter.lower())
            scale_factor = 1.8
    else:
        folder, letter_norm = get_folder_for_letter(letter)
        if letter_norm == "i":
            scale_factor = 1.2
        elif letter_norm == "ii" or letter_norm == "uu" or letter_norm == "eh" or letter_norm == "aa" or letter_norm == "aaa" or letter_norm == "aaaa" or letter_norm == "aaaaa" or letter_norm == "oo" or letter_norm == "ooo" or letter_norm == "ooooo" or letter_norm == "ee" or letter_norm == "cc":
            scale_factor = 1.6
        elif letter_norm == ",," or  letter_norm == "\'\'" or letter_norm == ":" or letter_norm == "," or letter_norm == ";" or letter_norm == "-" :
            scale_factor = 0.5
        elif letter_norm == "0" or letter_norm == "1" or letter_norm == "2" or letter_norm == "3" or letter_norm == "4" or letter_norm == "5" or letter_norm == "6" or letter_norm == "7" or letter_norm == "8" or letter_norm == "9" or letter_norm == "(" or letter_norm == ")" or letter_norm == "[" or letter_norm == "]" or letter_norm == "?" :
            scale_factor = 2.0
        elif letter_norm == "e":
            scale_factor = 1.0
        elif letter_norm in high_letters:
            scale_factor = 2.0
        elif letter_norm in descenders:
            scale_factor = 1.8
        else:
            scale_factor = 1.0

    if folder is None:
        return None, None

    files = glob(os.path.join(folder, "*.jpg"))
    if not files:
        return None, None

    file = random.choice(files)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(cv2.bitwise_not(img_bin))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        letter_img = img_bin[y:y+h, x:x+w]
    else:
        letter_img = img_bin

    scale = (target_height / float(letter_img.shape[0])) * scale_factor
    new_w = int(letter_img.shape[1] * scale)
    new_h = int(letter_img.shape[0] * scale)
    letter_img = cv2.resize(letter_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return letter_img, scale

def build_word_image(word, spacing=5):
    """
    Concatena as imagens dos caracteres de uma palavra, alinhando-as pela linha de base.
    """
    letter_images = []
    baselines = []

    # Processamento de cada caractere para determinar a linha base
    for char in word:
        if char == " ":
            space_width = int(target_height * 0.6)
            letter_img = np.ones((target_height, space_width), dtype=np.uint8) * 255
            scale = 1.0
        else:
            letter_to_use = choose_letter_for_folder(char)
            letter_img, scale = load_letter_image(letter_to_use)
            if letter_img is None:
                print(f"Amostra para o caractere '{char}' (usando '{letter_to_use}') não encontrada.")
                continue

        letter_images.append((char, letter_img))
        h = letter_img.shape[0]
        # Aplica o ajuste somente se for letra minúscula com descendente
        if char.islower() and char in descenders:
            base = int(h * 0.5)
        else:
            base = h
        baselines.append(base)

    if not letter_images:
        return None

    global_baseline = max(baselines)
    total_width = sum([img.shape[1] for _, img in letter_images]) + spacing * (len(letter_images) - 1)
    extra_above = max([img.shape[0] - base for (char, img), base in zip(letter_images, baselines)])
    canvas_height = global_baseline + extra_above

    canvas = np.ones((canvas_height, total_width), dtype=np.uint8) * 255
    current_x = 0
    for idx, (char, img) in enumerate(letter_images):
        h, w = img.shape
        # Novamente, só aplica o ajuste para minúsculas com descendente
        if char.islower() and char in descenders:
            base = int(h * 0.8)
            extra_shift = descender_shift
        else:
            base = h
            extra_shift = 0

        y_offset = global_baseline - base + extra_shift
        canvas[y_offset:y_offset+h, current_x:current_x+w] = img
        current_x += w + spacing

    return canvas

def wrap_text_into_lines(text, max_width, spacing=5):
    """
    Recebe o texto inteiro e quebra-o em linhas de modo que cada linha (montada com build_word_image)
    não ultrapasse 'max_width'. Retorna uma lista de strings (linhas).
    """
    wrapped_lines = []
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            # Linha vazia, por exemplo, para separar parágrafos
            wrapped_lines.append("")
            continue

        words = paragraph.split()
        current_line = words[0]
        for word in words[1:]:
            candidate_line = current_line + " " + word
            candidate_img = build_word_image(candidate_line)
            if candidate_img is None:
                # Se não for possível construir a imagem, ignore esta palavra
                continue
            # Se a imagem resultante exceder o max_width, finalize a linha atual
            if candidate_img.shape[1] > max_width:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line = candidate_line
        wrapped_lines.append(current_line)
    return wrapped_lines

def build_page_image(lines_img_list, a4_width, a4_height, margin_left, margin_right, margin_top, line_spacing):
    """
    Cria um canvas do tamanho A4 e posiciona as imagens das linhas a partir das margens.
    Se alguma linha ultrapassar a área de conteúdo, ela é recortada para caber.
    """
    canvas = np.ones((a4_height, a4_width), dtype=np.uint8) * 255
    y_offset = margin_top
    content_width = a4_width - margin_left - margin_right  # largura disponível para o texto
    
    for img in lines_img_list:
        h, w = img.shape
        
        # Recorta a linha se ultrapassar a largura disponível
        if w > content_width:
            img = img[:, :content_width]
            w = content_width
        
        # Posicionamento horizontal
        if margin_left + w > a4_width:
            w = a4_width - margin_left
            img = img[:, :w]
        
        canvas[y_offset:y_offset+h, margin_left:margin_left+w] = img
        y_offset += h + line_spacing
        
        if y_offset >= a4_height:
            break
            
    return canvas

def build_pages(text, a4_width=2480, a4_height=3508,
                margin_left=50, margin_right=50, margin_top=50, margin_bottom=50,
                line_spacing=15):
    """
    Realiza a quebra de linha e paginação:
      - O texto é quebrado em linhas (word wrapping) para não exceder a área de conteúdo.
      - Quando o conteúdo ultrapassa a área disponível verticalmente, uma nova página é iniciada.
    
    Retorna uma lista de imagens (páginas).
    """
    max_text_width = a4_width - margin_left - margin_right
    max_text_height = a4_height - margin_top - margin_bottom
    wrapped_lines = wrap_text_into_lines(text, max_text_width, spacing=line_spacing)
    
    pages = []
    current_page_lines = []
    current_page_height = 0

    for line in wrapped_lines:
        if line.strip() == "":
            line_img = np.ones((target_height, 10), dtype=np.uint8) * 255
        else:
            line_img = build_word_image(line)
            if line_img is None:
                continue

        line_h = line_img.shape[0]
        additional_height = line_h if current_page_height == 0 else line_spacing + line_h

        if current_page_height + additional_height > max_text_height:
            page = build_page_image(current_page_lines, a4_width, a4_height, margin_left, margin_right, margin_top, line_spacing)
            pages.append(page)
            current_page_lines = []
            current_page_height = 0

        current_page_lines.append(line_img)
        current_page_height += additional_height

    if current_page_lines:
        page = build_page_image(current_page_lines, a4_width, a4_height, margin_left, margin_right, margin_top, line_spacing)
        pages.append(page)

    return pages

# ------------------------- Exemplo de uso -------------------------

text = """
Era uma vez um homem chamado "Jailson Mendes". Ele estava comendo margarina Delícia

Below is a concise, technically rigorous response to each query:

Overfitting in Neural Networks:
Overfitting occurs when a network models the training data too closely, capturing noise and spurious patterns, which deteriorates its generalization performance on unseen data.

Techniques to Avoid Overfitting:
Common methods include regularization (L1, L2), dropout, early stopping based on validation loss, data augmentation, reducing model complexity, and employing cross-validation.

Learning Rate in a Neural Network:
The learning rate is a hyperparameter that controls the size of the weight updates during each iteration of gradient descent, influencing convergence speed and stability.

Adjusting the Learning Rate:
It can be tuned via learning rate schedules (step decay, exponential decay, cosine annealing), adaptive optimization algorithms (Adam, RMSprop), or hyperparameter optimization methods such as grid or random search.

Defining the Ideal Number of Hidden Layers:
This decision is problem-specific; it requires balancing model capacity and computational cost. Empirical experimentation, domain expertise, and techniques like cross-validation guide the selection.

"""

# Cria as páginas com o texto
pages = build_pages(text)

# Salva cada página como uma imagem separada
for idx, page_img in enumerate(pages):
    filename = f"page_{idx+1}.jpg"
    cv2.imwrite(filename, page_img)
    print(f"A página {idx+1} foi salva como '{filename}'.")
