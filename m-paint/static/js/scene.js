// Cena vetorial + undo/redo por comandos — puro, sem DOM.
//
// A cena é uma lista ordenada de objetos que é RE-DESENHADA (replay) na
// camada de tinta a cada mutação, em vez de guardar snapshots raster.
// Um snapshot RGBA de 1600×1000 custa ~6,4 MB *cada*; redesenhar algumas
// centenas de polylines custa < 10 ms. E de graça: redimensionar a janela
// não perde nada, e a borracha (destination-out) fica correta porque a
// ordem do replay preserva a ordem em que as coisas aconteceram.
//
// Desde a M3 os objetos vivem em COORDENADAS MATEMÁTICAS (o quadro é uma
// folha infinita; pan/zoom movem o desenho todo). As ferramentas trabalham
// em pixels; o Board converte com toMathObject() na hora do commit, e
// drawObject() transforma de volta a cada replay. Larguras de tinta são
// guardadas em unidades matemáticas — dar zoom amplia o traço, como uma
// lupa sobre papel. Curvas ('curve') são a exceção: guardam a expressão e
// são re-amostradas por repaint (plot.js), com largura fixa em px.
//
// Objetos:
//   { kind:'stroke', tool, color, width, composite, paths:[[{x,y},...],...] }
//   { kind:'stamp',  img, x, y, w, h, latex }        // x,y = canto sup. esq.
//   { kind:'wipe',   x, y, w, h }                    // OCR: apaga região
//   { kind:'curve',  form, spec, color, widthPx }    // viva, ver plot.js

import { sampleCurve } from "./plot.js";

export class Scene {
  constructor() {
    this.objects = [];
    this.undoStack = [];
    this.redoStack = [];
  }

  add(obj) {
    this.objects.push(obj);
    this.undoStack.push({ type: "add", obj });
    this.redoStack.length = 0;
  }

  clear() {
    if (!this.objects.length) return;
    this.undoStack.push({ type: "clear", removed: this.objects });
    this.objects = [];
    this.redoStack.length = 0;
  }

  undo() {
    const op = this.undoStack.pop();
    if (!op) return false;
    if (op.type === "add") {
      this.objects.pop();
    } else { // clear
      this.objects = op.removed;
    }
    this.redoStack.push(op);
    return true;
  }

  redo() {
    const op = this.redoStack.pop();
    if (!op) return false;
    if (op.type === "add") {
      this.objects.push(op.obj);
    } else {
      this.objects = [];
    }
    this.undoStack.push(op);
    return true;
  }

  get canUndo() { return this.undoStack.length > 0; }
  get canRedo() { return this.redoStack.length > 0; }
}

export function drawStrokeObj(ctx, s) {
  ctx.save();
  ctx.globalCompositeOperation = s.composite || "source-over";
  ctx.strokeStyle = s.color;
  ctx.lineWidth = s.width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const path of s.paths) {
    if (!path.length) continue;
    ctx.beginPath();
    ctx.moveTo(path[0].x, path[0].y);
    if (path.length === 1) {
      ctx.lineTo(path[0].x + 0.01, path[0].y); // clique parado vira um ponto
    }
    for (let i = 1; i < path.length; i++) {
      ctx.lineTo(path[i].x, path[i].y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

// ferramenta entrega pixels -> cena guarda matemática (chamado no commit)
export function toMathObject(o, vp) {
  if (o.kind === "stroke") {
    return {
      ...o,
      width: o.width / vp.pxPerUnit,
      paths: o.paths.map((path) => path.map((p) => vp.toMath(p.x, p.y))),
    };
  }
  if (o.kind === "stamp" || o.kind === "wipe") {
    const tl = vp.toMath(o.x, o.y); // canto superior esquerdo
    return { ...o, x: tl.x, y: tl.y, w: o.w / vp.pxPerUnit, h: o.h / vp.pxPerUnit };
  }
  return o; // curvas já nascem em termos matemáticos
}

export function drawObject(ctx, o, vp) {
  if (o.kind === "stroke") {
    drawStrokeObj(ctx, {
      ...o,
      // clamp: bem longe do zoom original o traço ainda é visível
      width: Math.max(0.35, o.width * vp.pxPerUnit),
      paths: o.paths.map((path) => path.map((p) => vp.toPx(p.x, p.y))),
    });
  } else if (o.kind === "stamp") {
    const p = vp.toPx(o.x, o.y);
    ctx.drawImage(o.img, p.x, p.y, o.w * vp.pxPerUnit, o.h * vp.pxPerUnit);
  } else if (o.kind === "wipe") {
    // usado pelo OCR ("apagar traços reconhecidos"): no replay limpa só o
    // que veio antes dele, então continua correto no meio do histórico
    const p = vp.toPx(o.x, o.y);
    ctx.clearRect(p.x, p.y, o.w * vp.pxPerUnit, o.h * vp.pxPerUnit);
  } else if (o.kind === "curve") {
    drawStrokeObj(ctx, {
      color: o.color,
      width: o.widthPx,
      composite: "source-over",
      paths: sampleCurve(o, vp),
    });
  }
}

export function renderScene(ctx, objects, vp) {
  for (const o of objects) drawObject(ctx, o, vp);
}
