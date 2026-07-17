// O quadro: 3 canvases empilhados + roteamento de eventos de ponteiro.
//
//   grid    (fundo)  — grade/eixos; redesenha só em pan/zoom/toggle
//   ink     (meio)   — conteúdo commitado; redesenhado por replay da cena
//   preview (topo)   — traço ao vivo / fantasma do carimbo; recebe os eventos
//
// Todos são dimensionados em devicePixelRatio (backing store = CSS px × dpr,
// contexto escalado uma vez), então todo o código de desenho fala em CSS px.

import { renderScene } from "./scene.js";
import { drawGrid } from "./viewport.js";

export class Board {
  constructor(canvases, scene, vp) {
    this.canvases = canvases; // {grid, ink, preview}
    this.gtx = canvases.grid.getContext("2d");
    this.itx = canvases.ink.getContext("2d");
    this.ptx = canvases.preview.getContext("2d");
    this.scene = scene;
    this.vp = vp;
    this.gridOn = true;
    this.tool = null;
    this.onSceneChange = () => {};
    this._dragging = false;
    this._panning = null;
    this._firstSize = true;
    this._bind();
  }

  resize() {
    const rect = this.canvases.grid.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    for (const c of Object.values(this.canvases)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
      c.style.width = w + "px";
      c.style.height = h + "px";
      c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    this.vp.resize(w, h);
    if (this._firstSize) {
      this.vp.setDefaultScale();
      this._firstSize = false;
    }
    this.repaintGrid();
    this.repaintInk(); // cena vetorial: nada se perde ao redimensionar
  }

  repaintGrid() {
    this.gtx.clearRect(0, 0, this.vp.w, this.vp.h);
    if (this.gridOn) drawGrid(this.gtx, this.vp);
  }

  repaintInk() {
    this.itx.clearRect(0, 0, this.vp.w, this.vp.h);
    renderScene(this.itx, this.scene.objects);
  }

  clearPreview() {
    this.ptx.clearRect(0, 0, this.vp.w, this.vp.h);
  }

  setTool(tool) {
    this.tool?.cancel?.();
    this.clearPreview();
    this.tool = tool;
    this.canvases.preview.style.cursor = tool?.cursor || "default";
  }

  _pt(e) {
    const r = this.canvases.preview.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  _bind() {
    const cv = this.canvases.preview;

    cv.addEventListener("pointerdown", (e) => {
      if (e.button === 1) { // botão do meio = pan da grade
        e.preventDefault();
        this._panning = { x: e.clientX, y: e.clientY };
        cv.setPointerCapture(e.pointerId);
        return;
      }
      if (e.button !== 0) return;
      cv.setPointerCapture(e.pointerId);
      this._dragging = true;
      this.tool?.onDown?.(this._pt(e), this);
    });

    cv.addEventListener("pointermove", (e) => {
      if (this._panning) {
        this.vp.panPx(e.clientX - this._panning.x, e.clientY - this._panning.y);
        this._panning = { x: e.clientX, y: e.clientY };
        this.repaintGrid();
        return;
      }
      const p = this._pt(e);
      if (this._dragging) this.tool?.onMove?.(p, this);
      else this.tool?.onHover?.(p, this);
    });

    const finish = (e) => {
      if (this._panning) { this._panning = null; return; }
      if (!this._dragging) return;
      this._dragging = false;
      const obj = this.tool?.onUp?.(this._pt(e), this);
      this.clearPreview();
      if (obj) this.scene.add(obj);
      this.repaintInk(); // sempre: a borracha desenha ao vivo na tinta
      if (obj) this.onSceneChange();
    };
    cv.addEventListener("pointerup", finish);
    cv.addEventListener("pointercancel", () => {
      this._panning = null;
      if (this._dragging) {
        this._dragging = false;
        this.tool?.cancel?.();
        this.clearPreview();
        this.repaintInk();
      }
    });

    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const p = this._pt(e);
      this.vp.zoomAt(p.x, p.y, e.deltaY < 0 ? 1.15 : 1 / 1.15);
      this.repaintGrid();
    }, { passive: false });

    cv.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  // Recorte da camada de tinta (para o OCR), na resolução do backing store.
  // Devolve só o base64 (sem o prefixo data:).
  cropInkPNG({ x, y, w, h }) {
    const dpr = window.devicePixelRatio || 1;
    const tmp = document.createElement("canvas");
    tmp.width = Math.max(1, Math.round(w * dpr));
    tmp.height = Math.max(1, Math.round(h * dpr));
    tmp.getContext("2d").drawImage(
      this.canvases.ink,
      Math.round(x * dpr), Math.round(y * dpr), tmp.width, tmp.height,
      0, 0, tmp.width, tmp.height,
    );
    return tmp.toDataURL("image/png").split(",")[1];
  }

  // PNG com fundo branco; grade opcional. Exporta na resolução do backing
  // store (nítido em telas hiDPI).
  exportPNG(includeGrid) {
    const dpr = window.devicePixelRatio || 1;
    const tmp = document.createElement("canvas");
    tmp.width = Math.round(this.vp.w * dpr);
    tmp.height = Math.round(this.vp.h * dpr);
    const ctx = tmp.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, this.vp.w, this.vp.h);
    if (includeGrid) drawGrid(ctx, this.vp);
    renderScene(ctx, this.scene.objects);
    return tmp.toDataURL("image/png");
  }
}
