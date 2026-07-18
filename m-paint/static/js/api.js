// Cliente da API do backend.

export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return [20, 24, 60];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Pede o dobro da altura de exibição (limitado a 512) e mostra na metade:
// em telas hiDPI o carimbo fica nítido em vez de borrado.
export async function renderLatex(latex, displayHeightPx, colorHex) {
  const heightPx = Math.min(512, displayHeightPx * 2);
  const res = await fetch("/api/render_latex", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ latex, height_px: heightPx, color: hexToRgb(colorHex) }),
  });
  if (!res.ok) throw new Error(`servidor respondeu ${res.status}`);
  const d = await res.json();
  const img = new Image();
  img.src = `data:image/png;base64,${d.png_base64}`;
  await img.decode();
  const scale = displayHeightPx / d.height;
  // rendered=false: matplotlib mathtext não entendeu o LaTeX (ex.:
  // \begin{array}) e caiu no fallback de texto literal -- não é uma
  // fórmula de verdade, é texto bruto com a mesma aparência de carimbo
  return { img, latex, w: d.width * scale, h: displayHeightPx, rendered: d.rendered };
}

// Recorte dos traços (base64 de PNG) -> LaTeX reconhecido pelo pix2tex.
// A primeira chamada é lenta: o backend carrega o modelo (~100 MB) na hora.
export async function ocrPng(pngBase64) {
  const res = await fetch("/api/ocr", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ png_base64: pngBase64 }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(typeof data.detail === "string"
      ? data.detail : `servidor respondeu ${res.status}`);
  }
  return data; // {latex, low_confidence}
}
