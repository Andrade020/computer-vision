// Bootstrap: instancia cena/viewport/board, registra ferramentas e liga a UI.

import { Scene } from "./scene.js";
import { Viewport } from "./viewport.js";
import { Board } from "./board.js";
import { makePenTool } from "./tools/pen.js";
import { makeManhattanTool } from "./tools/manhattan.js";
import { makeBrownianTool } from "./tools/brownian.js";
import { makeStampTool } from "./tools/stamp.js";
import { makeOcrTool } from "./tools/ocr.js";
import { plotExpression } from "./plot.js";
import { renderLatex, ocrPng } from "./api.js";
import { ExprError } from "./expr.js";

const $ = (id) => document.getElementById(id);

const scene = new Scene();
const vp = new Viewport();
const board = new Board(
  { grid: $("layer-grid"), ink: $("layer-ink"), preview: $("layer-preview") },
  scene, vp,
);

// ---- ferramentas -----------------------------------------------------------
const opts = {
  color: () => $("colorInput").value,
  width: () => +$("widthRange").value,
  sigma: () => +$("sigmaRange").value,
};
const stampState = { img: null, latex: "", w: 0, h: 0 };
const tools = {
  pen: makePenTool(opts),
  eraser: makePenTool(opts, { eraser: true }),
  manhattan: makeManhattanTool(opts),
  brownian: makeBrownianTool(opts),
  stamp: makeStampTool(stampState),
  ocr: makeOcrTool(handleOcrSelect),
};
const toolOrder = ["pen", "eraser", "manhattan", "brownian", "stamp", "ocr"];

function setTool(name) {
  board.setTool(tools[name]);
  document.querySelectorAll("[data-tool]").forEach((b) => {
    b.classList.toggle("active", b.dataset.tool === name);
  });
}
document.querySelectorAll("[data-tool]").forEach((b) => {
  b.addEventListener("click", () => setTool(b.dataset.tool));
});

// ---- sliders + labels ------------------------------------------------------
function bindSlider(rangeId, labelId, fmt = (v) => v) {
  const el = $(rangeId), label = $(labelId);
  const update = () => { label.textContent = fmt(el.value); };
  el.addEventListener("input", update);
  update();
}
bindSlider("widthRange", "widthVal", (v) => `${v} px`);
bindSlider("sigmaRange", "sigmaVal", (v) => `σ = ${(+v).toFixed(2)}`);
bindSlider("latexHeight", "latexHeightVal", (v) => `${v} px`);

// ---- histórico -------------------------------------------------------------
function refreshHistoryButtons() {
  $("undoBtn").disabled = !scene.canUndo;
  $("redoBtn").disabled = !scene.canRedo;
}
board.onSceneChange = refreshHistoryButtons;

function doUndo() { if (scene.undo()) { board.repaintInk(); refreshHistoryButtons(); } }
function doRedo() { if (scene.redo()) { board.repaintInk(); refreshHistoryButtons(); } }
$("undoBtn").addEventListener("click", doUndo);
$("redoBtn").addEventListener("click", doRedo);
$("clearBtn").addEventListener("click", () => {
  scene.clear();
  board.repaintInk();
  refreshHistoryButtons();
});

// ---- grade -----------------------------------------------------------------
$("gridToggle").addEventListener("change", (e) => {
  board.gridOn = e.target.checked;
  board.repaintGrid();
});

// ---- plot ------------------------------------------------------------------
function showError(el, msg) { el.textContent = msg; el.hidden = !msg; }

function doPlot() {
  const src = $("plotInput").value;
  try {
    const obj = plotExpression(src, vp, {
      color: opts.color(),
      width: Math.max(2, opts.width()),
    });
    scene.add(obj);
    board.repaintInk();
    refreshHistoryButtons();
    showError($("plotError"), "");
  } catch (err) {
    const msg = err instanceof ExprError && Number.isFinite(err.pos)
      ? `${err.message} (posição ${err.pos + 1})`
      : err.message;
    showError($("plotError"), msg);
  }
}
$("plotBtn").addEventListener("click", doPlot);
$("plotInput").addEventListener("keydown", (e) => { if (e.key === "Enter") doPlot(); });

// ---- carimbo LaTeX ---------------------------------------------------------
async function doRenderLatex() {
  const latex = $("latexInput").value.trim();
  if (!latex) { showError($("latexError"), "digite uma fórmula LaTeX"); return; }
  const btn = $("latexBtn");
  btn.disabled = true;
  btn.textContent = "renderizando…";
  try {
    const r = await renderLatex(latex, +$("latexHeight").value, opts.color());
    Object.assign(stampState, r);
    showError($("latexError"), "");
    setTool("stamp"); // fantasma segue o cursor; clique posiciona
  } catch (err) {
    showError($("latexError"), err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Renderizar";
  }
}
$("latexBtn").addEventListener("click", doRenderLatex);
$("latexInput").addEventListener("keydown", (e) => { if (e.key === "Enter") doRenderLatex(); });

// ---- OCR (desenhou a fórmula -> pix2tex -> LaTeX -> carimbo) ---------------
async function handleOcrSelect(rect) {
  const status = $("ocrStatus");
  showError($("ocrError"), "");
  status.hidden = false;
  status.textContent = "reconhecendo… (a primeira vez carrega o modelo, ~10 s)";
  try {
    const b64 = board.cropInkPNG(rect);
    const { latex } = await ocrPng(b64);
    $("latexInput").value = latex;
    if ($("ocrWipe").checked) {
      scene.add({ kind: "wipe", x: rect.x, y: rect.y, w: rect.w, h: rect.h });
      board.repaintInk();
      refreshHistoryButtons();
    }
    status.hidden = true;
    await doRenderLatex(); // renderiza bonito e ativa o carimbo
  } catch (err) {
    status.hidden = true;
    showError($("ocrError"), err.message);
  }
}

// ---- exportar --------------------------------------------------------------
$("exportBtn").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = board.exportPNG($("exportGrid").checked);
  a.download = "mathboard.png";
  a.click();
});

// ---- teclado ---------------------------------------------------------------
window.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|TEXTAREA)$/.test(e.target.tagName);
  if (e.ctrlKey && !typing && e.key.toLowerCase() === "z") {
    e.preventDefault();
    e.shiftKey ? doRedo() : doUndo();
    return;
  }
  if (e.ctrlKey && !typing && e.key.toLowerCase() === "y") {
    e.preventDefault();
    doRedo();
    return;
  }
  if (typing) return;
  if (e.key === "Escape") {
    board.tool?.cancel?.();
    board.clearPreview();
    return;
  }
  const idx = ["1", "2", "3", "4", "5", "6"].indexOf(e.key);
  if (idx >= 0) setTool(toolOrder[idx]);
});

// ---- tamanho ---------------------------------------------------------------
new ResizeObserver(() => board.resize())
  .observe($("layer-grid").parentElement);
board.resize();
setTool("pen");
refreshHistoryButtons();
