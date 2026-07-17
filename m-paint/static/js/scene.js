// Cena vetorial + undo/redo por comandos — puro, sem DOM.
//
// A cena é uma lista ordenada de objetos que é RE-DESENHADA (replay) na
// camada de tinta a cada mutação, em vez de guardar snapshots raster.
// Um snapshot RGBA de 1600×1000 custa ~6,4 MB *cada*; redesenhar algumas
// centenas de polylines custa < 10 ms. E de graça: redimensionar a janela
// não perde nada, e a borracha (destination-out) fica correta porque a
// ordem do replay preserva a ordem em que as coisas aconteceram.
//
// Objetos:
//   { kind:'stroke', tool, color, width, composite, paths:[[{x,y},...],...], expr? }
//   { kind:'stamp',  img, x, y, w, h, latex }

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

export function drawObject(ctx, o) {
  if (o.kind === "stroke") {
    drawStrokeObj(ctx, o);
  } else if (o.kind === "stamp") {
    ctx.drawImage(o.img, o.x, o.y, o.w, o.h);
  } else if (o.kind === "wipe") {
    // usado pelo OCR ("apagar traços reconhecidos"): no replay limpa só o
    // que veio antes dele, então continua correto no meio do histórico
    ctx.clearRect(o.x, o.y, o.w, o.h);
  }
}

export function renderScene(ctx, objects) {
  for (const o of objects) drawObject(ctx, o);
}
