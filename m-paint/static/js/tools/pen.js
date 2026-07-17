// Caneta livre. A borracha é a MESMA ferramenta com composite
// 'destination-out': no replay da cena ela "fura" só o que veio antes dela,
// exatamente como aconteceu ao vivo.

import { drawStrokeObj } from "../scene.js";

export function makePenTool(opts, { eraser = false } = {}) {
  let path = null;

  function obj() {
    return {
      kind: "stroke",
      tool: eraser ? "eraser" : "pen",
      color: eraser ? "#000" : opts.color(),
      width: opts.width() * (eraser ? 2.5 : 1),
      composite: eraser ? "destination-out" : "source-over",
      paths: [path],
    };
  }

  return {
    cursor: "crosshair",
    onDown(p) { path = [p]; },
    onMove(p, board) {
      if (!path) return;
      if (eraser) {
        // apaga ao vivo direto na tinta; o replay do commit dá o mesmo resultado
        const prev = path[path.length - 1];
        path.push(p);
        drawStrokeObj(board.itx, { ...obj(), paths: [[prev, p]] });
      } else {
        path.push(p);
        board.clearPreview();
        drawStrokeObj(board.ptx, obj());
      }
    },
    onUp(p, board) {
      if (!path) return null;
      path.push(p);
      const o = obj();
      path = null;
      return o;
    },
    cancel() { path = null; },
  };
}
