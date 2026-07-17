// Ferramenta de OCR: arraste um retângulo em volta da fórmula desenhada.
// A ferramenta só faz a SELEÇÃO (retângulo tracejado no preview) e entrega o
// rect ao callback; quem conversa com /api/ocr é o main.js. Não adiciona
// nenhum objeto à cena — por isso onUp devolve null.

function norm(a, b) {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    w: Math.abs(b.x - a.x),
    h: Math.abs(b.y - a.y),
  };
}

export function makeOcrTool(onSelect) {
  let start = null;

  function draw(rect, board) {
    board.clearPreview();
    const ctx = board.ptx;
    ctx.save();
    ctx.fillStyle = "rgba(108, 92, 231, 0.08)";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.strokeStyle = "#6c5ce7";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    ctx.restore();
  }

  return {
    cursor: "crosshair",
    onDown(p) { start = { ...p }; },
    onMove(p, board) { if (start) draw(norm(start, p), board); },
    onUp(p, board) {
      if (!start) return null;
      const rect = norm(start, p);
      start = null;
      board.clearPreview();
      if (rect.w >= 8 && rect.h >= 8) onSelect(rect);
      return null;
    },
    cancel() { start = null; },
  };
}
