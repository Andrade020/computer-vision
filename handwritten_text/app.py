import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import cv2
import os
import numpy as np
from PIL import Image
import os
import numpy as np
import random
from glob import glob
import unicodedata
import sys
import re

# =============================================================================
# ------------------------- funcs auxlrs -------------------------
##########################################################################

def normalize_for_filesystem(letter, form="NFC"):
    return unicodedata.normalize(form, letter)
############################################################################################

def get_folder_for_letter(letter, prefix=""):
    for form in ("NFC", "NFD"):
        letter_norm = unicodedata.normalize(form, letter)
        folder_path = os.path.join(data_dir, f"{prefix}{letter_norm}")
        if os.path.exists(folder_path) and glob(os.path.join(folder_path, "*.jpg")):
            return folder_path, letter_norm
    return None, None
##########################################################################################

def remove_accents(char):
    if char in ['ç', 'Ç']:
        return char
    nfkd = unicodedata.normalize('NFKD', char)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])
##############################################################

data_dir = "handwritten_text/segmented/recognized"
##########################################################################

high_letters = set("htflkbd")
descenders = set("jgqpçy")
########################################################################################

target_height = 46  
descender_shift = int(45 * (target_height / 100.0))
#############################################################################################

def choose_letter_for_folder(letter):
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
################################################################################################
    
    if letter in accent_mapping:
        letter_mapped = accent_mapping[letter]
    else:
        letter_mapped = letter
#####################################################################

    if letter_mapped == " ":
        return letter_mapped
############################################################################

    if letter_mapped.isupper():
        folder, letter_norm = get_folder_for_letter(letter_mapped, prefix="_")
        if folder is not None:
            return letter_norm
        else:
            return letter_mapped
    else:
        folder, letter_norm = get_folder_for_letter(letter_mapped)
        if folder is not None:
            return letter_norm
        else:
            return letter_mapped
###################################################################

def load_letter_image(letter):
    if letter.isupper():
        folder, letter_norm = get_folder_for_letter(letter, prefix="_")
        if folder is not None:
            scale_factor = 1.8
        else:
            folder, letter_norm = get_folder_for_letter(letter.lower())
            scale_factor = 1.8
    else:
        folder, letter_norm = get_folder_for_letter(letter)
        if letter_norm == "i":
            scale_factor = 1.4
        elif letter_norm in ("ii", "uu", "eh", "aa", "aaa", "aaaa", "aaaaa", "oo", "ooo", "ooooo", "ee", "cc"):
            scale_factor = 1.6
        elif letter_norm in (",," , ":", ",", ";"):
            scale_factor = 0.4
        elif letter_norm == "-":
            scale_factor = 0.9
        elif letter_norm == "y":
            scale_factor = 1.8
        elif letter_norm in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", "[", "]", "?"):
            scale_factor = 2.0
        elif letter_norm == "e":
            scale_factor = 1.0
        elif letter_norm in high_letters:
            scale_factor = 2.0
        elif letter_norm in descenders:
            scale_factor = 1.8
        else:
            scale_factor = 1.0
##########################################################################

    if folder is None:
        return None, None
########################################################################

    files = glob(os.path.join(folder, "*.jpg"))
    if not files:
        return None, None
#############################################################################################

    file = random.choice(files)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
##########################################################################

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#########################################################################

    ret, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(cv2.bitwise_not(img_bin))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        letter_img = img_bin[y:y+h, x:x+w]
    else:
        letter_img = img_bin
###################################################################################################

    scale = (target_height / float(letter_img.shape[0])) * scale_factor
    new_w = int(letter_img.shape[1] * scale)
    new_h = int(letter_img.shape[0] * scale)
    letter_img = cv2.resize(letter_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
###########################################################################################
    
    return letter_img, scale
###############################################################################

# ------------------------- simlc de tinta com cors flexvs -------------------------

def simulate_ink(gray_img, ink="blue", pixelation_factor=4):
    """
    #convrt uma imgm em escla de cinza (fundo brnco e trcs em prto) em uma
    #imgm colrd, simlnd a tinta da canta bic com o efto desjd.
    #############################################################################
    #parmtr:
      - gray_img: imagem em escala de cinza.
      - ink: cor desejada ("blue", "red", "green", "black").
      - pixelation_factor: controle para o efeito de pixelation.
    #################################################################################
    #retrna:
      - Imagem colorida (BGR) com a simulação da tinta.
    """
    colored = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    
    mask = gray_img < 240
    stroke_bin = np.uint8(mask) * 255
    dist = cv2.distanceTransform(stroke_bin, cv2.DIST_L2, 5)
    max_dist = dist[mask].max() if np.any(mask) else 1.0
    dist_norm = dist / (max_dist + 1e-6)
    
    noise_scale = 16
    noise_shape = (gray_img.shape[0] // noise_scale + 1, gray_img.shape[1] // noise_scale + 1)
    small_noise = np.random.uniform(-1, 1, noise_shape).astype(np.float32)
    noise_low_freq = cv2.resize(small_noise, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    noise_low_freq = cv2.GaussianBlur(noise_low_freq, (21, 21), 0)
    
    noise_mapped = (noise_low_freq + 1) / 2.0
    noise_min, noise_max = 0.2, 0.6
    noise_mapped = noise_mapped * (noise_max - noise_min) + noise_min
    
    T = 0.5
    alpha = np.clip((dist_norm - T) / (1.0 - T), 0, 1)
    combined_factor = (1 - alpha) * dist_norm + alpha * noise_mapped
    combined_factor = np.clip(combined_factor * 0.9, 0, 1)
    
    intensity = (combined_factor * 255).astype(np.uint8)
    
    if ink.lower() == "blue":
        colored[mask, 0] = intensity[mask]
        colored[mask, 1] = 0
        colored[mask, 2] = 0
    elif ink.lower() == "red":
        colored[mask, 2] = intensity[mask]
        colored[mask, 0] = 0
        colored[mask, 1] = 0
    elif ink.lower() == "green":
        colored[mask, 1] = intensity[mask]
        colored[mask, 0] = 0
        colored[mask, 2] = 0
    elif ink.lower() == "black":
        colored[mask] = np.stack([intensity[mask]]*3, axis=-1)
    else:
        colored[mask, 0] = intensity[mask]
        colored[mask, 1] = 0
        colored[mask, 2] = 0
        
    h, w = colored.shape[:2]
    downscaled = cv2.resize(colored, (w // pixelation_factor, h // pixelation_factor), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return pixelated

# ------------------------- Processamento de Texto com Comandos de Cor -------------------------

def parse_colored_characters(text):
    """
    procss o texto para identf comnds de cor no formto /cor{texto} e retrna
    uma lista de tupls (carctr, cor). o padro eh "blue".
    """    
    pattern = re.compile(r"/(\w+)\{(.*?)\}")
    result = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            segment = text[pos:m.start()]
            for ch in segment:
                result.append((ch, "blue"))
        color = m.group(1).lower()
        segment = m.group(2)
        for ch in segment:
            result.append((ch, color))
        pos = m.end()
    if pos < len(text):
        segment = text[pos:]
        for ch in segment:
            result.append((ch, "blue"))
    return result

def split_into_words(parsed_chars):
    """
    divde a lista de (carctr, cor) em palvrs, mantnd os espcs.
    """
    words = []
    current_word = []
    for ch, color in parsed_chars:
        if ch == " ":
            if current_word:
                words.append(current_word)
                current_word = []
            words.append([(" ", "blue")])
        else:
            current_word.append((ch, color))
    if current_word:
        words.append(current_word)
    return words

def build_word_image_colored(char_color_list, spacing=5):
    """
    Constrói a imagem de uma palavra a partir de uma lista de tuplas (caractere, cor).
    """
    letter_images = []
    baselines = []

    for ch, color in char_color_list:
        if ch == " ":
            space_width = int(target_height * 0.6)
            letter_img = np.ones((target_height, space_width, 3), dtype=np.uint8) * 255
        else:
            letter_to_use = choose_letter_for_folder(ch)
            gray_letter, scale = load_letter_image(letter_to_use)
            if gray_letter is None:
                print(f"Amostra para o caractere '{ch}' (usando '{letter_to_use}') não encontrada.")
                continue
            letter_img = simulate_ink(gray_letter, ink=color)
        letter_images.append((ch, letter_img))
        h = letter_img.shape[0]
        if ch.islower() and ch in descenders:
            base = int(h * 0.5)
        else:
            base = h
        baselines.append(base)

    if not letter_images:
        return None

    global_baseline = max(baselines)
    total_width = sum([img.shape[1] for _, img in letter_images]) + spacing * (len(letter_images) - 1)
    extra_above = max([img.shape[0] - base for (ch, img), base in zip(letter_images, baselines)])
    canvas_height = global_baseline + extra_above

    canvas = np.ones((canvas_height, total_width, 3), dtype=np.uint8) * 255
    current_x = 0
    for idx, (ch, img) in enumerate(letter_images):
        h, w, _ = img.shape
        if ch.islower() and ch in descenders:
            base = int(h * 0.8)
            extra_shift = descender_shift
        else:
            base = h
            extra_shift = 0
        y_offset = global_baseline - base + extra_shift
        canvas[y_offset:y_offset+h, current_x:current_x+w] = img
        current_x += w + spacing

    return canvas

def build_line_image(text_line, spacing=5):
    """
    Constrói a imagem de uma linha de texto que pode conter comandos de cor inline.
    """
    parsed = parse_colored_characters(text_line)
    return build_word_image_colored(parsed, spacing=spacing)

def wrap_text_into_lines_colored(text, max_width, spacing=5):
    """
    Divide o texto em linhas (word wrapping) respeitando o máximo em pixels, considerando os comandos de cor.
    """
    wrapped_lines = []
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            wrapped_lines.append("")
            continue
        parsed = parse_colored_characters(paragraph)
        words = split_into_words(parsed)
        
        current_line = []
        for word in words:
            candidate_line = current_line + word
            candidate_img = build_word_image_colored(candidate_line, spacing=spacing)
            if candidate_img is None:
                continue
            if candidate_img.shape[1] > max_width:
                if current_line:
                    wrapped_lines.append(current_line)
                    current_line = word
                else:
                    wrapped_lines.append(word)
                    current_line = []
            else:
                current_line = candidate_line
        if current_line:
            wrapped_lines.append(current_line)
    return wrapped_lines

def build_page_image_colored(lines_char_lists, a4_width, a4_height, margin_left, margin_right, margin_top, line_spacing):
    canvas = np.ones((a4_height, a4_width, 3), dtype=np.uint8) * 255
    y_offset = margin_top
    content_width = a4_width - margin_left - margin_right
    
    for char_list in lines_char_lists:
        line_img = build_word_image_colored(char_list, spacing=line_spacing)
        if line_img is None:
            continue
        h, w, _ = line_img.shape
        if w > content_width:
            line_img = line_img[:, :content_width]
            w = content_width
        if margin_left + w > a4_width:
            w = a4_width - margin_left
            line_img = line_img[:, :w]
        canvas[y_offset:y_offset+h, margin_left:margin_left+w] = line_img
        y_offset += h + line_spacing
        if y_offset >= a4_height:
            break
    return canvas

def build_pages_colored(text, a4_width=2480, a4_height=3508,
                        margin_left=50, margin_right=50, margin_top=50, margin_bottom=50,
                        line_spacing=15):
    max_text_width = a4_width - margin_left - margin_right
    max_text_height = a4_height - margin_top - margin_bottom
    wrapped_lines = wrap_text_into_lines_colored(text, max_text_width, spacing=line_spacing)
    
    pages = []
    current_page_lines = []
    current_page_height = 0

    for char_list in wrapped_lines:
        if char_list == "":
            line_img = np.ones((target_height, 10, 3), dtype=np.uint8) * 255
        else:
            line_img = build_word_image_colored(char_list, spacing=line_spacing)
            if line_img is None:
                continue

        line_h = line_img.shape[0]
        additional_height = line_h if current_page_height == 0 else line_spacing + line_h

        if current_page_height + additional_height > max_text_height:
            page = build_page_image_colored(current_page_lines, a4_width, a4_height, margin_left, margin_right, margin_top, line_spacing)
            pages.append(page)
            current_page_lines = []
            current_page_height = 0

        current_page_lines.append(char_list)
        current_page_height += additional_height

    if current_page_lines:
        page = build_page_image_colored(current_page_lines, a4_width, a4_height, margin_left, margin_right, margin_top, line_spacing)
        pages.append(page)

    return pages

# ------------------------- Simulação de Dobras no Papel -------------------------

def simulate_paper_folds(page_img, 
                         warp_amplitude=5, 
                         warp_period=200,
                         max_crease_intensity=0.012,
                         crease_count=5, 
                         crease_shadow_width=30,
                         global_noise_scale=0.05, 
                         brightness_variation=0.1, 
                         texture_strength=0.05):
    """
    Aplica efeitos de dobras e distorções em toda a página, simulando o aspecto
    de um papel sulfite escaneado com dobras distribuídas aleatoriamente.
    
    Para cada dobra, sorteia-se:
      - Uma posição central aleatória.
      - Um ângulo e um comprimento aleatórios.
      - A intensidade da sombra, variando de 0 (imperceptível) até max_crease_intensity.
      - O formato: "retangular" ou "triangular" (com largura decrescente nas extremidades).
    
    Parâmetros:
      - page_img: imagem (BGR) da página.
      - warp_amplitude: amplitude da distorção global em pixels.
      - warp_period: período da distorção senoidal.
      - max_crease_intensity: valor máximo de intensidade para uma dobra (0 a 1).
      - crease_count: número de dobras a serem geradas.
      - crease_shadow_width: valor base para a largura da sombra de cada dobra.
      - global_noise_scale: escala do ruído global para imperfeições.
      - brightness_variation: variação da iluminação (gradiente vertical).
      - texture_strength: força da textura que simula imperfeições do papel.
      
    Retorna:
      - Imagem (BGR) final com os efeitos aplicados.
    """
    import cv2
    import numpy as np
    import random

    H, W = page_img.shape[:2]
    
    # 1. Distorção Global (Warp) com função senoidal
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    phase = np.random.uniform(0, 2 * np.pi)
    offset = warp_amplitude * np.sin(2 * np.pi * y / warp_period + phase)
    map_x = x + offset
    map_y = y
    warped_img = cv2.remap(page_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_float = warped_img.astype(np.float32) / 255.0
    
    # 2. Criação da máscara de dobras (creases) com formatos variados
    crease_mask = np.zeros((H, W), dtype=np.float32)
    # Define comprimento mínimo e máximo para o traçado da dobra
    min_length = 0.2 * min(W, H)
    max_length = 1.2 * max(W, H)
    for i in range(crease_count):
        # Posição central aleatória
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)
        # Ângulo aleatório (em radianos)
        theta = random.uniform(0, 2 * np.pi)
        # Comprimento aleatório da dobra
        L = random.uniform(min_length, max_length)
        # Calcula os endpoints da linha, centrada em (cx, cy)
        x1 = int(cx - np.cos(theta) * L / 2)
        y1 = int(cy - np.sin(theta) * L / 2)
        x2 = int(cx + np.cos(theta) * L / 2)
        y2 = int(cy + np.sin(theta) * L / 2)
        # Sorteia a intensidade da dobra (pode ser 0, para ficar imperceptível, até o máximo)
        crease_val = random.uniform(0.0, max_crease_intensity)
        # Sorteia o formato da dobra: retangular ou triangular
        crease_shape = random.choice(["rectangular", "triangular"])
        
        # Para o formato retangular, usamos cv2.line
        if crease_shape == "rectangular":
            local_width = random.uniform(0.5 * crease_shadow_width, 1.5 * crease_shadow_width)
            local_width = int(local_width)
            if local_width % 2 == 0:
                local_width += 1
            temp_mask = np.zeros((H, W), dtype=np.float32)
            cv2.line(temp_mask, (x1, y1), (x2, y2), color=crease_val, thickness=local_width)
            crease_mask = np.maximum(crease_mask, temp_mask)
        else:
            # Para o formato triangular, constrói-se um polígono em forma de losango
            # que tem os endpoints com espessura zero e largura máxima no centro.
            # Calcula o vetor perpendicular normalizado à linha
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if d == 0:
                p = (0, 0)
            else:
                p = (-(y2 - y1) / d, (x2 - x1) / d)
            # Define uma largura máxima para a dobra (na região central)
            local_width = random.uniform(0.5 * crease_shadow_width, 1.5 * crease_shadow_width)
            # Calcula o ponto médio
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            # Define os vértices:
            # A: endpoint 1, B: ponto médio deslocado para um lado, 
            # C: endpoint 2, D: ponto médio deslocado para o lado oposto.
            A = (x1, y1)
            C = (x2, y2)
            B = (int(mx + p[0] * (local_width / 2)), int(my + p[1] * (local_width / 2)))
            D = (int(mx - p[0] * (local_width / 2)), int(my - p[1] * (local_width / 2)))
            pts = np.array([A, B, C, D], dtype=np.int32)
            temp_mask = np.zeros((H, W), dtype=np.float32)
            cv2.fillConvexPoly(temp_mask, pts, crease_val)
            crease_mask = np.maximum(crease_mask, temp_mask)
    
    # Suaviza a máscara para criar transições graduais entre as dobras e o fundo
    crease_mask = cv2.GaussianBlur(crease_mask, (31, 31), 0)
    crease_mask = np.expand_dims(crease_mask, axis=2)
    
    # Aplica a redução de brilho nas áreas das dobras
    warped_with_crease = warped_float * (1 - crease_mask)
    
    # 3. Ruído global de baixa frequência para simular imperfeições
    noise_small = np.random.uniform(-1, 1, (H // 20 + 1, W // 20 + 1, 1)).astype(np.float32)
    noise_low = cv2.resize(noise_small, (W, H), interpolation=cv2.INTER_LINEAR)
    noise_low = cv2.GaussianBlur(noise_low, (21, 21), 0)
    if noise_low.ndim == 2:
        noise_low = np.expand_dims(noise_low, axis=2)
    warped_with_noise = warped_with_crease + global_noise_scale * noise_low
    warped_with_noise = np.clip(warped_with_noise, 0, 1)
    
    # 4. Variação de brilho com gradiente vertical para simular iluminação irregular
    gradient = np.tile(np.linspace(1 - brightness_variation, 1 + brightness_variation, H)[:, None], (1, W))
    gradient = np.expand_dims(gradient, axis=2)
    warped_final = warped_with_noise * gradient
    warped_final = np.clip(warped_final, 0, 1)
    
    # 5. Adiciona textura de papel para imperfeições adicionais
    texture_noise = np.random.normal(loc=0, scale=texture_strength, size=(H, W, 3)).astype(np.float32)
    warped_textured = warped_final + texture_noise
    warped_textured = np.clip(warped_textured, 0, 1)
    
    final_img = (warped_textured * 255).astype(np.uint8)
    return final_img

# =============================================================================
def save_images_to_pdf(images, pdf_path):
    """
    Converte uma lista de imagens (em formato OpenCV, BGR) para um PDF usando o PIL.
    """
    pil_images = []
    for img in images:
        # Converter de BGR para RGB e depois para objeto PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_images.append(pil_img)
    if pil_images:
        pil_images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=pil_images[1:])

def process_text_and_generate_pdf(text, aplicar_dobras, aplicar_textura, pdf_path, progress_callback):
    """
    Processa o texto, gera as páginas (com ou sem dobras e texturas conforme os parâmetros)
    e salva o resultado em PDF.
    """
    try:
        # Monta as páginas a partir do texto inserido
        pages = build_pages_colored(text)
        final_pages = []
        total = len(pages)
        for idx, page_img in enumerate(pages):
            progress_callback(f"Processando página {idx+1} de {total}...")
            if aplicar_dobras or aplicar_textura:
                page_with_effect = simulate_paper_folds(
                    page_img,
                    warp_amplitude=5 if aplicar_textura else 0,
                    warp_period=200,
                    max_crease_intensity=0.2 if aplicar_dobras else 0,
                    crease_count=5 if aplicar_dobras else 0,
                    crease_shadow_width=30,
                    global_noise_scale=0.05 if aplicar_textura else 0,
                    brightness_variation=0.05,
                    texture_strength=0.05 if aplicar_textura else 0
                )
            else:
                page_with_effect = page_img
            final_pages.append(page_with_effect)
        progress_callback("Salvando PDF...")
        save_images_to_pdf(final_pages, pdf_path)
        progress_callback("Concluído")
    except Exception as exc:
        progress_callback(f"Erro: {str(exc)}")
        raise

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gerador de PDF - Handwritten Text")
        self.geometry("800x600")
        self.create_widgets()
        
    def create_widgets(self):
        # Editor de texto
        self.text_editor = tk.Text(self, wrap="word", height=15)
        self.text_editor.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Frame para as opções (checkboxes)
        options_frame = tk.Frame(self)
        options_frame.pack(padx=10, pady=5, fill="x")
        
        # Checkbox para simular dobras
        self.var_dobras = tk.BooleanVar(value=True)
        chk_dobras = tk.Checkbutton(options_frame, text="Simular Dobras", variable=self.var_dobras)
        chk_dobras.pack(side="left", padx=5)
        
        # Checkbox para aplicar textura
        self.var_textura = tk.BooleanVar(value=True)
        chk_textura = tk.Checkbutton(options_frame, text="Aplicar Textura", variable=self.var_textura)
        chk_textura.pack(side="left", padx=5)
        
        # Botão para gerar o PDF
        self.btn_generate = tk.Button(self, text="Gerar PDF", command=self.start_generation)
        self.btn_generate.pack(pady=10)
        
        # Label para mensagens de status
        self.lbl_status = tk.Label(self, text="Status: Aguardando")
        self.lbl_status.pack(pady=5)
        
        # Cria a barra de progresso, mas não a exibe inicialmente
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        
    def start_generation(self):
        # Pega o texto do editor
        text = self.text_editor.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Aviso", "Digite o texto a ser processado.")
            return
        
        # Permite que o usuário escolha onde salvar o PDF
        pdf_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if not pdf_path:
            return
        
        # Desabilita o botão para evitar cliques múltiplos durante o processamento
        self.btn_generate.config(state="disabled")
        
        # Exibe a barra de progresso e inicia-a
        self.progress.pack(fill="x", padx=10, pady=5)
        self.progress.start(10)
        self.lbl_status.config(text="Processando...")
        
        # Cria uma thread para rodar o processamento sem travar a interface
        thread = threading.Thread(target=self.run_generation, args=(
            text, self.var_dobras.get(), self.var_textura.get(), pdf_path
        ))
        thread.start()
        
    def run_generation(self, text, aplicar_dobras, aplicar_textura, pdf_path):
        def update_status(message):
            # Atualiza a label de status na interface
            self.lbl_status.config(text=message)
        try:
            process_text_and_generate_pdf(text, aplicar_dobras, aplicar_textura, pdf_path, update_status)
            # Ao concluir, mostra uma mensagem de sucesso
            self.after(0, lambda msg=self.lbl_status.cget("text"): messagebox.showinfo("Sucesso", f"PDF salvo em {pdf_path}"))
        except Exception as exc:
            # Captura a exceção e garante que seu valor seja passado para a lambda
            error_msg = str(exc)
            self.after(0, lambda err=error_msg: messagebox.showerror("Erro", err))
        finally:
            self.after(0, self.generation_done)
            
    def generation_done(self):
        # Para a barra de progresso e a remove da interface; reabilita o botão
        self.progress.stop()
        self.progress.pack_forget()
        self.lbl_status.config(text="Concluído")
        self.btn_generate.config(state="normal")
        
if __name__ == "__main__":
    app = Application()
    app.mainloop()