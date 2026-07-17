// Cliente da API do backend. Em M2 ganha ocr(pngBlob) -> latex.

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
  return { img, latex, w: d.width * scale, h: displayHeightPx };
}
