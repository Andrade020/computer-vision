// Modo "+": traços restritos a horizontal/vertical (desenho Manhattan).
//
// Máquina de estados com histerese:
//  - deadzone: nenhum eixo é escolhido até o cursor andar > 6 px do início;
//    aí o eixo dominante do movimento vence.
//  - extensão: no eixo H o endpoint vivo é (x do cursor, y do trilho) — o
//    cursor é PROJETADO no trilho atual.
//  - troca: só quando o desvio perpendicular passa de 12 px commitamos um
//    canto e viramos para o outro eixo. Sem essa histerese, movimento quase
//    diagonal viraria uma escadinha trêmula trocando de eixo a cada pixel.

import { drawStrokeObj } from "../scene.js";

export class ManhattanTracker {
  constructor(deadzone = 6, switchThreshold = 12) {
    this.deadzone = deadzone;
    this.switchThreshold = switchThreshold;
  }

  begin(p) {
    this.start = { ...p };
    this.corners = [{ ...p }];
    this.anchor = { ...p };
    this.axis = null;
  }

  // devolve a polyline atual (cantos + endpoint vivo)
  update(p) {
    if (this.axis === null) {
      const dx = p.x - this.start.x, dy = p.y - this.start.y;
      if (Math.hypot(dx, dy) < this.deadzone) return [...this.corners];
      this.axis = Math.abs(dx) >= Math.abs(dy) ? "h" : "v";
    }
    let live;
    if (this.axis === "h") {
      live = { x: p.x, y: this.anchor.y };
      if (Math.abs(p.y - this.anchor.y) > this.switchThreshold) {
        this.corners.push(live);
        this.anchor = live;
        this.axis = "v";
        live = { x: this.anchor.x, y: p.y };
      }
    } else {
      live = { x: this.anchor.x, y: p.y };
      if (Math.abs(p.x - this.anchor.x) > this.switchThreshold) {
        this.corners.push(live);
        this.anchor = live;
        this.axis = "h";
        live = { x: p.x, y: this.anchor.y };
      }
    }
    return [...this.corners, live];
  }
}

export function makeManhattanTool(opts) {
  const tracker = new ManhattanTracker();
  let active = false;
  let pts = null;

  const obj = () => ({
    kind: "stroke", tool: "manhattan",
    color: opts.color(), width: opts.width(),
    composite: "source-over", paths: [pts],
  });

  return {
    cursor: "crosshair",
    onDown(p) { tracker.begin(p); pts = [{ ...p }]; active = true; },
    onMove(p, board) {
      if (!active) return;
      pts = tracker.update(p);
      board.clearPreview();
      drawStrokeObj(board.ptx, obj());
    },
    onUp(p) {
      if (!active) return null;
      pts = tracker.update(p);
      active = false;
      return pts.length > 1 ? obj() : null;
    },
    cancel() { active = false; pts = null; },
  };
}
