// Curvas "vivas": o objeto guarda a EXPRESSÃO, não os pontos.
// A cada repaint a curva é re-amostrada para o viewport atual — por isso
// pan/zoom nunca revelam amostragem antiga: a curva é sempre re-desenhada
// lisa, como no GeoGebra. (A tinta, ao contrário, é congelada em coordenadas
// matemáticas e amplia como desenho.)
//
// Três formas:
//   cartesian:  y = f(x)      — 1 amostra a cada ~2 px de coluna do viewport
//   parametric: x(t), y(t)    — N amostras em t ∈ [t0, t1]
//   polar:      r(t)          — idem, com (x,y) = r·(cos t, sin t)
//
// Amostras não finitas, pontos muito além do viewport ou SALTOS grandes em
// pixels (assíntotas: tan(x), r = 1/t...) quebram a polyline em ramos.

import { compile } from "./expr.js";

export function makeCurve(form, spec, { color = "#6c5ce7", widthPx = 2.5 } = {}) {
  compileCurve(form, spec); // valida já: lança ExprError com posição
  return { kind: "curve", form, spec: { ...spec }, color, widthPx };
}

function compileCurve(form, spec) {
  if (form === "cartesian") return { f: compile(spec.fx, "x") };
  if (form === "parametric") return { x: compile(spec.x, "t"), y: compile(spec.y, "t") };
  if (form === "polar") return { r: compile(spec.r, "t") };
  throw new Error(`forma de curva desconhecida: ${form}`);
}

function fns(o) {
  return (o._fns ??= compileCurve(o.form, o.spec)); // cache por objeto
}

export function sampleCurve(o, vp) {
  const paths = [];
  let cur = null;
  let last = null;
  const jumpLimit = (vp.w + vp.h) / 2;   // salto maior que isso = assíntota
  const xHalf = vp.w / 2 / vp.pxPerUnit;
  const yHalf = vp.h / 2 / vp.pxPerUnit;

  const flush = () => {
    if (cur && cur.length > 1) paths.push(cur);
    cur = null;
    last = null;
  };
  const push = (mx, my) => {
    if (!Number.isFinite(mx) || !Number.isFinite(my)
      || Math.abs(mx - vp.cx) > 10 * xHalf || Math.abs(my - vp.cy) > 10 * yHalf) {
      flush();
      return;
    }
    const p = vp.toPx(mx, my);
    if (last && Math.hypot(p.x - last.x, p.y - last.y) > jumpLimit) flush();
    (cur ??= []).push(p);
    last = p;
  };

  if (o.form === "cartesian") {
    const { f } = fns(o);
    for (let px = 0; px <= vp.w; px += 2) {
      const mx = vp.toMath(px, 0).x;
      push(mx, f(mx));
    }
  } else {
    const N = 1200;
    const { t0, t1 } = o.spec;
    if (o.form === "parametric") {
      const { x, y } = fns(o);
      for (let k = 0; k <= N; k++) {
        const t = t0 + (t1 - t0) * k / N;
        push(x(t), y(t));
      }
    } else { // polar
      const { r } = fns(o);
      for (let k = 0; k <= N; k++) {
        const t = t0 + (t1 - t0) * k / N;
        const rv = r(t);
        push(rv * Math.cos(t), rv * Math.sin(t));
      }
    }
  }
  flush();
  return paths;
}
