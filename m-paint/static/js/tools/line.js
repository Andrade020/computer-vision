// Ferramenta de linha (H/V): clique em A, arraste até B, solte -- traça uma
// reta perfeitamente horizontal ou vertical (o eixo de maior deslocamento
// vence, recalculado a cada movimento, então dá pra "girar" a prévia entre
// H e V antes de soltar). Uma régua, não um traçador de gesto.
//
// Substitui o antigo "modo +" (perseguição de eixo com histerese em tempo
// real): aquele exigia controlar a velocidade/direção do gesto do mouse com
// cuidado pra não zigzaguear; isso aqui é clicar, mirar, soltar.

import { drawStrokeObj } from "../scene.js";

// pura e testável: dado o ponto inicial e o cursor atual, devolve o
// endpoint travado no eixo dominante (>= empata para horizontal)
export function lineEndpoint(start, p) {
  const dx = p.x - start.x, dy = p.y - start.y;
  return Math.abs(dx) >= Math.abs(dy)
    ? { x: p.x, y: start.y }
    : { x: start.x, y: p.y };
}

export function makeLineTool(opts) {
  let start = null;

  const obj = (end) => ({
    kind: "stroke", tool: "line",
    color: opts.color(), width: opts.width(),
    composite: "source-over", paths: [[start, end]],
  });

  return {
    cursor: "crosshair",
    onDown(p) { start = { ...p }; },
    onMove(p, board) {
      if (!start) return;
      board.clearPreview();
      drawStrokeObj(board.ptx, obj(lineEndpoint(start, p)));
    },
    onUp(p) {
      if (!start) return null;
      const o = obj(lineEndpoint(start, p));
      start = null;
      return o;
    },
    cancel() { start = null; },
  };
}
