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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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


# mounted last so /api/* routes win
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
