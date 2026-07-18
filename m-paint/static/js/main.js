// Bootstrap: instancia cena/viewport/board, registra ferramentas e liga a UI.

import { Scene, toMathObject } from "./scene.js";
import { Viewport } from "./viewport.js";
import { Board } from "./board.js";
import { makePenTool } from "./tools/pen.js";
import { makeManhattanTool } from "./tools/manhattan.js";
import { makeBrownianTool } from "./tools/brownian.js";
import { makeStampTool } from "./tools/stamp.js";
import { makeOcrTool } from "./tools/ocr.js";
import { makeHandTool } from "./tools/hand.js";
import { makeCurve } from "./plot.js";
import { renderLatex, ocrPng } from "./api.js";
import { ExprError, evalConst } from "./expr.js";

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
  hand: makeHandTool(),
};
const toolOrder = ["pen", "eraser", "manhattan", "brownian", "stamp", "ocr", "hand"];

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

const plotMode = $("plotMode");
function updatePlotRows() {
  const m = plotMode.value;
  $("rowCartesian").hidden = m !== "cartesian";
  $("rowParametric").hidden = m !== "parametric";
  $("rowPolar").hidden = m !== "polar";
  $("rowRange").hidden = m === "cartesian";
}
plotMode.addEventListener("change", updatePlotRows);
updatePlotRows();

function doPlot() {
  try {
    const m = plotMode.value;
    let spec;
    if (m === "cartesian") {
      spec = { fx: $("plotInput").value };
    } else {
      // as faixas aceitam expressões constantes: "2pi", "pi/2", "-3"
      const t0 = evalConst($("tMin").value);
      const t1 = evalConst($("tMax").value);
      spec = m === "parametric"
        ? { x: $("plotX").value, y: $("plotY").value, t0, t1 }
        : { r: $("plotR").value, t0, t1 };
    }
    const curve = makeCurve(m, spec, {
      color: opts.color(),
      widthPx: Math.max(2, opts.width()),
    });
    scene.add(curve); // curvas já nascem matemáticas: sem conversão
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
for (const id of ["plotInput", "plotX", "plotY", "plotR", "tMin", "tMax"]) {
  $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") doPlot(); });
}

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
    // rendered=false: mathtext não entendeu o LaTeX (ex.: \begin{array}, que
    // o pix2tex às vezes aluciona) e o carimbo é texto bruto, não fórmula
    showError($("latexError"), r.rendered ? "" :
      "⚠ isso não é LaTeX que o renderizador entende — o carimbo vai mostrar texto bruto, não uma fórmula.");
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
  const spinner = $("ocrSpinner");
  const statusText = $("ocrStatusText");
  showError($("ocrError"), "");
  status.classList.remove("warn");
  status.hidden = false;
  spinner.hidden = false;
  statusText.textContent = "reconhecendo… (a primeira vez carrega o modelo, ~10 s)";
  try {
    const b64 = board.cropInkPNG(rect);
    const { latex, low_confidence } = await ocrPng(b64);
    spinner.hidden = true;
    $("latexInput").value = latex;
    // seleção "simples demais" (poucos traços) -> não apaga o desenho
    // original, mesmo com a opção marcada: o resultado pode ser lixo, e
    // sumir com o rabisco junto seria perder trabalho por nada
    if ($("ocrWipe").checked && !low_confidence) {
      scene.add(toMathObject({ kind: "wipe", ...rect }, vp));
      board.repaintInk();
      refreshHistoryButtons();
    }
    status.classList.toggle("warn", low_confidence);
    if (low_confidence) {
      status.hidden = false;
      statusText.textContent = "⚠ seleção simples demais, ou o modelo não devolveu algo " +
        "reconhecível como fórmula — confira o resultado com atenção ou digite direto.";
    } else {
      status.hidden = true;
    }
    await doRenderLatex(); // renderiza bonito e ativa o carimbo
  } catch (err) {
    spinner.hidden = true;
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
    setTool("hand"); // sai do modo de escrita atual (caneta, carimbo...) para a mãozinha
    return;
  }
  const idx = ["1", "2", "3", "4", "5", "6", "7"].indexOf(e.key);
  if (idx >= 0) setTool(toolOrder[idx]);
});

// ---- tamanho ---------------------------------------------------------------
new ResizeObserver(() => board.resize())
  .observe($("layer-grid").parentElement);
board.resize();
setTool("pen");
refreshHistoryButtons();
