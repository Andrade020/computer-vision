// Linha browniana: o usuário arrasta uma reta A→B e ela vira uma PONTE
// browniana — um passeio aleatório condicionado a terminar exatamente em B.
//
// Construção clássica: dado o passeio S_k = Σ g_i·σ·√(Δt) com g_i ~ N(0,1),
//   bridge_k = S_k − (k/N)·S_N
// zera as duas pontas. O desvio é aplicado na direção NORMAL ao segmento AB,
// então a "reta" serpenteia em torno do caminho reto. A escala σ·√L deixa o
// slider de volatilidade invariante ao comprimento do traço.
//
// Seed fixa por arraste (nova a cada pointerdown): o preview estica
// elasticamente em vez de ferver com ruído novo a cada pointermove.

import { drawStrokeObj } from "../scene.js";
import { mulberry32, gaussians, randomSeed } from "../rng.js";

export function brownianBridge(A, B, seed, sigma) {
  const dx = B.x - A.x, dy = B.y - A.y;
  const L = Math.hypot(dx, dy);
  if (L < 1e-6) return [{ ...A }, { ...B }];

  const N = Math.max(16, Math.min(512, Math.round(L / 3)));
  const g = gaussians(mulberry32(seed));
  const s = sigma * Math.sqrt(L / N);

  const S = [0];
  for (let k = 1; k <= N; k++) S.push(S[k - 1] + g() * s);

  const nx = -dy / L, ny = dx / L; // normal unitária de AB
  const pts = [];
  for (let k = 0; k <= N; k++) {
    const t = k / N;
    const b = S[k] - t * S[N]; // ponte: pontas fixas
    pts.push({ x: A.x + dx * t + nx * b, y: A.y + dy * t + ny * b });
  }
  return pts;
}

export function makeBrownianTool(opts) {
  let A = null, seed = 0, pts = null;

  const obj = () => ({
    kind: "stroke", tool: "brownian",
    color: opts.color(), width: opts.width(),
    composite: "source-over", paths: [pts],
  });

  return {
    cursor: "crosshair",
    onDown(p) { A = { ...p }; seed = randomSeed(); pts = null; },
    onMove(p, board) {
      if (!A) return;
      pts = brownianBridge(A, p, seed, opts.sigma());
      board.clearPreview();
      drawStrokeObj(board.ptx, obj());
    },
    onUp(p) {
      if (!A) return null;
      pts = brownianBridge(A, p, seed, opts.sigma());
      A = null;
      const o = pts.length > 1 ? obj() : null;
      pts = null;
      return o;
    },
    cancel() { A = null; pts = null; },
  };
}
