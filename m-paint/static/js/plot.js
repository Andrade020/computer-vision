// Amostragem de y = f(x) sobre o viewport -> objeto stroke da cena.
//
// Uma amostra a cada ~2 px de coluna. Amostras não finitas (1/0, log(-1))
// ou que estouram 10× a meia-altura do viewport QUEBRAM a polyline: é isso
// que faz 1/x virar dois ramos em vez de um espigão vertical na assíntota.
//
// A curva vira um stroke comum em coordenadas de PIXEL (simplificação da M1:
// um único espaço de coordenadas para tinta e plots; pan/zoom re-posiciona
// apenas a grade e plots futuros). Guardamos `expr` no objeto para o futuro.

import { compile } from "./expr.js";

export function plotExpression(src, vp, { color = "#6c5ce7", width = 2.5 } = {}) {
  const f = compile(src); // lança ExprError com {message, pos} se inválida
  const yHalf = vp.h / 2 / vp.pxPerUnit;
  const yLimit = 10 * yHalf;

  const paths = [];
  let cur = null;
  for (let px = 0; px <= vp.w; px += 2) {
    const mx = vp.toMath(px, 0).x;
    const my = f(mx);
    if (!Number.isFinite(my) || Math.abs(my - vp.cy) > yLimit) {
      if (cur && cur.length > 1) paths.push(cur);
      cur = null;
      continue;
    }
    (cur ??= []).push(vp.toPx(mx, my));
  }
  if (cur && cur.length > 1) paths.push(cur);

  if (!paths.length) {
    throw new Error("nenhum ponto da curva cai no viewport atual");
  }
  return {
    kind: "stroke", tool: "plot", expr: src,
    color, width, composite: "source-over", paths,
  };
}
