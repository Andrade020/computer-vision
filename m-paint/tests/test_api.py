import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.app import app

client = TestClient(app)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_render_valid_latex():
    r = client.post("/api/render_latex",
                    json={"latex": r"\frac{\sqrt{x}}{2}", "height_px": 64})
    assert r.status_code == 200
    d = r.json()
    png = base64.b64decode(d["png_base64"])
    assert png[:8] == PNG_MAGIC
    assert d["width"] > 0
    assert d["height"] == 64


def test_render_custom_color_and_height():
    r = client.post("/api/render_latex",
                    json={"latex": "x^2", "height_px": 128,
                          "color": [200, 30, 30]})
    assert r.status_code == 200
    assert r.json()["height"] == 128


def test_bad_latex_falls_back_to_literal():
    r = client.post("/api/render_latex", json={"latex": r"\notacommand{{{"})
    assert r.status_code == 200
    png = base64.b64decode(r.json()["png_base64"])
    assert png[:8] == PNG_MAGIC


def test_empty_latex_is_422():
    r = client.post("/api/render_latex", json={"latex": ""})
    assert r.status_code == 422


def test_height_out_of_range_is_422():
    r = client.post("/api/render_latex", json={"latex": "x", "height_px": 9000})
    assert r.status_code == 422


def test_index_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "MathBoard" in r.text


# ---- OCR (M2) ---------------------------------------------------------------

def _png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_cleanup_latex():
    from server.ocr import cleanup_latex
    assert cleanup_latex(r"\scriptstyle{\frac{x}{2}}") == r"\frac{x}{2}"
    assert cleanup_latex(r"\displaystyle \scriptstyle x") == "x"
    assert cleanup_latex("{a+b}") == "a+b"
    assert cleanup_latex("{a}+{b}") == "{a}+{b}"  # chaves não englobam tudo


def test_ocr_invalid_png_is_422():
    r = client.post("/api/ocr", json={"png_base64": "bm90cG5n"})
    assert r.status_code == 422


def test_ocr_empty_selection_is_422():
    pytest.importorskip("pix2tex")
    blank = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    r = client.post("/api/ocr", json={"png_base64": _png_b64(blank)})
    assert r.status_code == 422


def test_ocr_roundtrip_recognizes_rendered_formula():
    # integração: renderiza com o mathimg e reconhece com o pix2tex
    # (carrega o modelo -> lento na primeira vez, alguns segundos depois)
    pytest.importorskip("pix2tex")
    from server.mathimg import render_math
    img, _ = render_math(r"\frac{x^2+1}{2}", 120)
    r = client.post("/api/ocr", json={"png_base64": _png_b64(img)})
    assert r.status_code == 200
    latex = r.json()["latex"]
    assert "frac" in latex and "x" in latex
