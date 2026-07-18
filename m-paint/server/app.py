"""
MathBoard (m-paint) backend.

FastAPI app with two jobs:
  1. POST /api/render_latex -- rasterize a LaTeX math string to a transparent
     PNG (matplotlib mathtext, offline) and return it as base64 + dimensions.
     This is the render path the M2 OCR feature (/api/ocr, pix2tex) will feed.
  2. Serve the static frontend from ../static.

Run from the m-paint directory:
    uvicorn server.app:app --reload --port 8000
"""
import base64
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .mathimg import render_math

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="MathBoard", version="0.1.0")


class RenderReq(BaseModel):
    latex: str = Field(min_length=1)
    height_px: int = Field(default=64, ge=8, le=512)
    color: tuple[int, int, int] = (20, 24, 60)


@app.post("/api/render_latex")
def render_latex(req: RenderReq):
    ink = tuple(max(0, min(255, c)) for c in req.color)
    # render_math falls back to \mathrm literal text on mathtext parse errors,
    # so bad LaTeX still yields a 200 with a readable render
    img, width = render_math(req.latex, req.height_px, ink=ink)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "width": width,
        "height": img.height,
    }


class OcrReq(BaseModel):
    png_base64: str = Field(min_length=1)


@app.post("/api/ocr")
def ocr(req: OcrReq):
    """Recorte dos traços (PNG base64) -> LaTeX reconhecido (pix2tex)."""
    try:
        from .ocr import EmptySelection, ocr_image
    except ImportError:
        raise HTTPException(503, "pix2tex não instalado (pip install pix2tex)")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(req.png_base64)))
        img.load()
    except Exception:
        raise HTTPException(422, "PNG inválido")
    try:
        latex, low_confidence = ocr_image(img)
    except EmptySelection as e:
        raise HTTPException(422, str(e))
    if not latex:
        raise HTTPException(422, "o modelo não reconheceu nada na seleção")
    return {"latex": latex, "low_confidence": low_confidence}


# mounted last so /api/* routes win
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
