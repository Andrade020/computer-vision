// Mãozinha: navega (pan) sem desenhar nada. É o destino do Esc a partir de
// qualquer outra ferramenta -- uma forma rápida de "sair do modo de
// escrita" (caneta, carimbo, etc.) sem precisar acertar um botão pequeno.
// Reaproveita vp.panPx, a mesma matemática do pan por botão do meio em
// board.js, então o comportamento é idêntico independente de como se ativa.

export function makeHandTool() {
  let last = null;

  return {
    cursor: "grab",
    onDown(p, board) {
      last = p;
      board.canvases.preview.style.cursor = "grabbing";
    },
    onMove(p, board) {
      if (!last) return;
      board.vp.panPx(p.x - last.x, p.y - last.y);
      last = p;
      board.repaintAll();
    },
    onUp(_p, board) {
      last = null;
      board.canvases.preview.style.cursor = "grab";
      return null; // não cria objeto de cena
    },
    cancel() { last = null; },
  };
}
