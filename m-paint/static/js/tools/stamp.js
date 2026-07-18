// Carimbo LaTeX: depois que o painel renderiza a fórmula (via backend),
// esta ferramenta mostra um fantasma semi-transparente seguindo o cursor e
// o clique posiciona um objeto 'stamp' na cena.
//
// Por padrão funciona uma vez só: depois de carimbar, `state.img` é
// zerado, o fantasma some e cliques seguintes não fazem nada até a pessoa
// clicar "Renderizar" de novo -- evita carimbar a mesma fórmula sem querer
// várias vezes. Com `opts.multi()` marcado, o carimbo continua carregado
// (comportamento antigo: clique quantas vezes quiser).

export function makeStampTool(state, opts) {
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
      const stamped = {
        kind: "stamp",
        img: state.img,
        x: p.x - state.w / 2,
        y: p.y - state.h / 2,
        w: state.w,
        h: state.h,
        latex: state.latex,
      };
      if (!opts.multi()) state.img = null; // descarrega: só carimba uma vez
      return stamped;
    },
    cancel() {},
  };
}
