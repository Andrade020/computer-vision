// Mapeamento matemática ↔ pixels + desenho da grade — puro, sem DOM
// (recebe um ctx 2D já pronto, então é testável com um canvas offscreen).
//
// Modelo: {cx, cy} são as coordenadas matemáticas do CENTRO do canvas e
// pxPerUnit é a escala uniforme. y matemático cresce para cima; y de pixel
// cresce para baixo — daí os sinais trocados em toPx/toMath.

export class Viewport {
  constructor() {
    this.cx = 0;
    this.cy = 0;
    this.pxPerUnit = 60;
    this.w = 800;
    this.h = 600;
  }

  resize(w, h) {
    this.w = w;
    this.h = h;
  }

  setDefaultScale() {
    // x cobre [-10, 10]; y segue a proporção do canvas
    this.pxPerUnit = this.w / 20;
  }

  toPx(mx, my) {
    return {
      x: this.w / 2 + (mx - this.cx) * this.pxPerUnit,
      y: this.h / 2 - (my - this.cy) * this.pxPerUnit,
    };
  }

  toMath(px, py) {
    return {
      x: this.cx + (px - this.w / 2) / this.pxPerUnit,
      y: this.cy - (py - this.h / 2) / this.pxPerUnit,
    };
  }

  zoomAt(px, py, factor) {
    const before = this.toMath(px, py);
    this.pxPerUnit = Math.min(2000, Math.max(2, this.pxPerUnit * factor));
    const after = this.toMath(px, py);
    // mantém o ponto sob o cursor fixo
    this.cx += before.x - after.x;
    this.cy += before.y - after.y;
  }

  panPx(dx, dy) {
    this.cx -= dx / this.pxPerUnit;
    this.cy += dy / this.pxPerUnit;
  }
}

// passo de tick 1-2-5: menor valor da forma {1,2,5}·10^k cuja distância em
// pixels fique >= minPx
export function tickStep(pxPerUnit, minPx = 48) {
  const target = minPx / pxPerUnit;
  const pow = Math.pow(10, Math.floor(Math.log10(target)));
  for (const m of [1, 2, 5, 10]) {
    if (pow * m >= target) return pow * m;
  }
  return pow * 10;
}

function formatTick(v) {
  // corta ruído de ponto flutuante (0.30000000000000004 -> "0.3")
  return String(parseFloat(v.toFixed(10)));
}

const COLORS = {
  minor: "#eceef3",
  major: "#d5d9e4",
  axis: "#7a8199",
  label: "#7a8199",
};

export function drawGrid(ctx, vp) {
  const step = tickStep(vp.pxPerUnit);
  const minor = step / 5;
  const { x: xMin, y: yMax } = vp.toMath(0, 0);
  const { x: xMax, y: yMin } = vp.toMath(vp.w, vp.h);
  const eps = minor / 1e6;

  const isMultiple = (v, s) => Math.abs(v / s - Math.round(v / s)) < 1e-6;

  // linhas verticais
  for (let v = Math.ceil(xMin / minor) * minor; v <= xMax + eps; v += minor) {
    const { x } = vp.toPx(v, 0);
    const major = isMultiple(v, step);
    ctx.strokeStyle = major ? COLORS.major : COLORS.minor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, vp.h);
    ctx.stroke();
  }
  // linhas horizontais
  for (let v = Math.ceil(yMin / minor) * minor; v <= yMax + eps; v += minor) {
    const { y } = vp.toPx(0, v);
    const major = isMultiple(v, step);
    ctx.strokeStyle = major ? COLORS.major : COLORS.minor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(vp.w, y);
    ctx.stroke();
  }

  // eixos
  const origin = vp.toPx(0, 0);
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1.6;
  if (origin.y >= 0 && origin.y <= vp.h) {
    ctx.beginPath();
    ctx.moveTo(0, origin.y);
    ctx.lineTo(vp.w, origin.y);
    ctx.stroke();
  }
  if (origin.x >= 0 && origin.x <= vp.w) {
    ctx.beginPath();
    ctx.moveTo(origin.x, 0);
    ctx.lineTo(origin.x, vp.h);
    ctx.stroke();
  }

  // labels nos ticks maiores, ao longo dos eixos (quando visíveis)
  ctx.fillStyle = COLORS.label;
  ctx.font = "11px system-ui, sans-serif";
  const axisYVisible = origin.y >= 0 && origin.y <= vp.h;
  const axisXVisible = origin.x >= 0 && origin.x <= vp.w;

  if (axisYVisible) {
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const ly = Math.min(vp.h - 14, origin.y + 4);
    for (let v = Math.ceil(xMin / step) * step; v <= xMax + eps; v += step) {
      if (Math.abs(v) < step / 2) continue; // 0 fica no canto da origem
      ctx.fillText(formatTick(v), vp.toPx(v, 0).x, ly);
    }
  }
  if (axisXVisible) {
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const lx = Math.max(24, origin.x - 5);
    for (let v = Math.ceil(yMin / step) * step; v <= yMax + eps; v += step) {
      if (Math.abs(v) < step / 2) continue;
      ctx.fillText(formatTick(v), lx, vp.toPx(0, v).y);
    }
  }
  if (axisXVisible && axisYVisible) {
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText("0", origin.x - 4, origin.y + 4);
  }
}
