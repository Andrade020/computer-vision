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
"""
import threading

from PIL import Image, ImageOps

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


def ocr_image(img):
    """PIL image (recorte do canvas) -> string LaTeX."""
    prepared = prepare_for_ocr(img)
    return cleanup_latex(get_model()(prepared))
