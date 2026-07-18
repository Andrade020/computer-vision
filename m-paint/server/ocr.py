"""
OCR de formulas: imagem de tracos -> string LaTeX, via pix2tex (LaTeX-OCR).

O modelo ViT (~65 MB) e baixado na primeira execucao e carregado
preguicosamente na primeira requisicao -- subir o servidor continua
instantaneo. Roda em CPU.

pix2tex foi treinado em formulas IMPRESSAS (renders de LaTeX), nao em
manuscrito. prepare_for_ocr() aproxima o desenho do usuario desse dominio:
funde os tracos coloridos sobre fundo branco, estica o contraste para a
tinta mais escura virar preto (autocontrast funciona para QUALQUER cor de
caneta), recorta no bounding box e devolve uma margem branca generosa.

O modelo tambem e sistematicamente ruim em expressoes triviais (uma letra,
"2x", "x+1") -- ver o comentario de ink_complexity() abaixo para o porque e
os dados de calibracao.
"""
import os
import threading

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from .mathimg import render_math

# o albumentations (dependência do pix2tex) checa atualização na rede ao
# importar; sem rede isso pode travar a primeira requisição de OCR
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

_model = None
_lock = threading.Lock()


class EmptySelection(ValueError):
    pass


def get_model():
    global _model
    with _lock:
        if _model is None:
            from pix2tex.cli import LatexOCR
            _model = LatexOCR()
        return _model


def prepare_for_ocr(img):
    """Traços do canvas (RGBA, fundo transparente) -> estilo 'impresso'."""
    white = Image.new("RGB", img.size, "white")
    if img.mode == "RGBA":
        white.paste(img, mask=img.getchannel("A"))
    else:
        white.paste(img.convert("RGB"))
    g = white.convert("L")

    bbox = g.point(lambda v: 255 if v < 250 else 0).getbbox()
    if bbox is None:
        raise EmptySelection("a seleção não contém traços")
    g = g.crop(bbox)

    g = ImageOps.autocontrast(g)
    return ImageOps.expand(g, border=24, fill=255).convert("RGB")


# O modelo costuma prefixar o resultado com comandos de estilo
# (\scriptstyle etc.) quando a entrada é pequena — ruído, não conteúdo.
_STYLES = ("\\scriptscriptstyle", "\\scriptstyle", "\\displaystyle", "\\textstyle")


def cleanup_latex(s):
    s = s.strip()
    changed = True
    while changed:
        changed = False
        for st in _STYLES:
            if s.startswith(st):
                s = s[len(st):].strip()
                changed = True
        if s.startswith("{") and s.endswith("}"):
            # desembrulha só se as chaves externas fecham no fim
            depth = 0
            whole = True
            for i, c in enumerate(s):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0 and i < len(s) - 1:
                        whole = False
                        break
            if whole:
                s = s[1:-1].strip()
                changed = True
    return s


# pix2tex foi treinado em recortes de artigos científicos, onde uma
# expressão isolada tipo "2x" quase nunca aparece sozinha como a fórmula
# inteira -- o modelo não tem calibração para "isto é trivial" e aluciona
# LaTeX complexo em vez de admitir incerteza (ver README, seção de
# limitações). Não dá para consertar isso pré-processando a imagem (testei
# reescalar seleções pequenas: ajudava casos triviais mas piorava fórmulas
# que já funcionavam bem) -- mas dá pra DETECTAR quando a entrada é simples
# demais para confiar no resultado, contando componentes de tinta
# "significativos" (filtrando manchas de anti-aliasing menores que
# MIN_COMPONENT_PX). Calibrado contra os próprios casos de teste: "2", "x",
# "2x", "x+1", "2x+1" (todos viram lixo ou erram letra) ficam em 1-4
# componentes; "x^2+2x+1", "\frac{a+b}{c}", "\int_0^1 x\,dx" etc. (todos
# corretos) ficam em 5+.
INK_THRESHOLD = 180
MIN_COMPONENT_PX = 4
LOW_CONFIDENCE_MAX_COMPONENTS = 4


def ink_complexity(img):
    """Nº de traços/símbolos distintos na imagem (heurística, não OCR)."""
    white = Image.new("L", img.size, 255)
    if img.mode == "RGBA":
        white.paste(img.convert("L"), mask=img.getchannel("A"))
    else:
        white.paste(img.convert("L"))
    mask = np.array(white) < INK_THRESHOLD
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return 0
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return int((sizes >= MIN_COMPONENT_PX).sum())


def ocr_image(img):
    """PIL image (recorte do canvas) -> (latex, low_confidence).

    low_confidence combina dois sinais independentes:
      - entrada simples demais (ink_complexity, ver acima);
      - o LaTeX reconhecido nem sequer é válido para o NOSSO renderizador
        (mathtext não entende \\begin{array}/\\begin{matrix} etc.). Isso pega
        exatamente o caso em que o modelo aluciona uma estrutura complexa
        (tabela, matriz) para um traço ambíguo: o resultado então nem chega
        a virar fórmula no carimbo, vira texto colado ilegível -- mais
        enganoso que um erro comum, porque parece que "renderizou" algo.
    """
    prepared = prepare_for_ocr(img)
    latex = cleanup_latex(get_model()(prepared))
    simple_input = ink_complexity(img) <= LOW_CONFIDENCE_MAX_COMPONENTS
    _, _, rendered_ok = render_math(latex, 64)
    low_confidence = simple_input or not rendered_ok
    return latex, low_confidence
