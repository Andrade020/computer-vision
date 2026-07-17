// Carimbo LaTeX: depois que o painel renderiza a fórmula (via backend),
// esta ferramenta mostra um fantasma semi-transparente seguindo o cursor e
// cada clique posiciona um objeto 'stamp' na cena. Em M2 o OCR alimenta o
// mesmo caminho: traços → /api/ocr → LaTeX → este carimbo.

export function makeStampTool(state) {
  function ghost(p, board) {
    if (!state.img) return;
    board.clearPreview();
    const ctx = board.ptx;
    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.drawImage(state.img, p.x - state.w / 2, p.y - state.h / 2, state.w, state.h);
    ctx.restore();
  }

  return {
    cursor: "copy",
    onHover(p, board) { ghost(p, board); },
    onDown(p, board) { ghost(p, board); },
    onMove(p, board) { ghost(p, board); },
    onUp(p, board) {
      if (!state.img) return null;
      board.clearPreview();
      return {
        kind: "stamp",
        img: state.img,
        x: p.x - state.w / 2,
        y: p.y - state.h / 2,
        w: state.w,
        h: state.h,
        latex: state.latex,
      };
    },
    cancel() {},
  };
}
