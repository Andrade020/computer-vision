// Compasso: clique pra FIXAR o centro (o "pino"), depois clique de novo
// pra DESENHAR um círculo até onde você clicou (o "lápis"). O pino continua
// fixo depois — clique de novo em qualquer lugar pra traçar outro círculo
// concêntrico, exatamente como um compasso de verdade deixa você riscar
// vários raios sem tirar a ponta do papel. Pra recomeçar em outro centro,
// clique de novo na ferramenta na barra (ou Esc, que sai pra mãozinha).
//
// Interação por CLIQUES independentes (down+up sem arrastar), não por um
// arraste único: o primeiro clique só pina; entre um clique e outro o braço
// do compasso (linha pino->cursor) e a prévia do círculo seguem o mouse via
// onHover, sem precisar segurar o botão.

import { drawStrokeObj } from "../scene.js";

const dist = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);

// pura e testável: N+1 pontos ao redor do centro, com raio r (o último
// ponto repete o primeiro, fechando o laço no replay de drawStrokeObj)
export function circlePoints(center, r, n = 96) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const th = (i / n) * Math.PI * 2;
    pts.push({ x: center.x + r * Math.cos(th), y: center.y + r * Math.sin(th) });
  }
  return pts;
}

export function makeCompassTool(opts) {
  let pin = null;

  function circleObj(center, r) {
    return {
      kind: "stroke", tool: "compass",
      color: opts.color(), width: opts.width(),
      composite: "source-over", paths: [circlePoints(center, r)],
    };
  }

  function preview(p, board) {
    board.clearPreview();
    const ctx = board.ptx;
    if (!pin) {
      // mira indicando "clique aqui pra fixar o centro"
      ctx.save();
      ctx.strokeStyle = opts.color();
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(p.x - 6, p.y); ctx.lineTo(p.x + 6, p.y);
      ctx.moveTo(p.x, p.y - 6); ctx.lineTo(p.x, p.y + 6);
      ctx.stroke();
      ctx.restore();
      return;
    }
    const r = dist(pin, p);
    // braço do compasso: pino -> cursor
    ctx.save();
    ctx.strokeStyle = opts.color();
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = Math.max(1, opts.width() / 2);
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pin.x, pin.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.restore();
    // prévia do círculo (cor cheia, pra ver o traço de verdade)
    drawStrokeObj(ctx, circleObj(pin, r));
    // marca o pino
    ctx.save();
    ctx.fillStyle = opts.color();
    ctx.beginPath();
    ctx.arc(pin.x, pin.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  return {
    cursor: "crosshair",
    onHover(p, board) { preview(p, board); },
    onMove(p, board) { preview(p, board); },
    onDown() {},
    onUp(p) {
      if (!pin) { pin = { ...p }; return null; } // 1º clique: só fixa o pino
      const r = dist(pin, p);
      if (r < 2) return null; // clique quase em cima do pino: ignora
      return circleObj(pin, r); // pino continua fixo pro próximo círculo
    },
    cancel() { pin = null; },
  };
}
