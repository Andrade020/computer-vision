// Bootstrap: instancia cena/viewport/board, registra ferramentas e liga a UI.

import { Scene } from "./scene.js";
import { Viewport } from "./viewport.js";
import { Board } from "./board.js";
import { makePenTool } from "./tools/pen.js";
import { makeLineTool } from "./tools/line.js";
import { makeBrownianTool } from "./tools/brownian.js";
import { makeCompassTool } from "./tools/compass.js";
import { makeStampTool } from "./tools/stamp.js";
import { makeHandTool } from "./tools/hand.js";
import { makeCurve } from "./plot.js";
import { renderLatex } from "./api.js";
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
  multi: () => $("stampMulti").checked,
};
const stampState = { img: null, latex: "", w: 0, h: 0 };
const tools = {
  pen: makePenTool(opts),
  eraser: makePenTool(opts, { eraser: true }),
  line: makeLineTool(opts),
  brownian: makeBrownianTool(opts),
  compass: makeCompassTool(opts),
  stamp: makeStampTool(stampState, opts),
  hand: makeHandTool(),
};
const toolOrder = ["pen", "eraser", "line", "brownian", "compass", "stamp", "hand"];

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

// ---- paleta de símbolos ------------------------------------------------------
// clicar insere o LaTeX no campo de fórmula na posição do cursor (não só no
// fim) -- selectionStart/End sobrevivem ao input perder o foco, então dá
// pra clicar/posicionar o cursor no texto, clicar num símbolo, repetir.
const SYMBOLS = [
  { sym: "π", tex: "\\pi" }, { sym: "θ", tex: "\\theta" },
  { sym: "α", tex: "\\alpha" }, { sym: "β", tex: "\\beta" },
  { sym: "λ", tex: "\\lambda" }, { sym: "μ", tex: "\\mu" },
  { sym: "σ", tex: "\\sigma" }, { sym: "Δ", tex: "\\Delta" },
  { sym: "∑", tex: "\\sum" }, { sym: "∫", tex: "\\int" },
  { sym: "√", tex: "\\sqrt{}", caret: 6 }, { sym: "∂", tex: "\\partial" },
  { sym: "±", tex: "\\pm" }, { sym: "×", tex: "\\times" },
  { sym: "÷", tex: "\\div" }, { sym: "≤", tex: "\\leq" },
  { sym: "≥", tex: "\\geq" }, { sym: "≠", tex: "\\neq" },
  { sym: "≈", tex: "\\approx" }, { sym: "∞", tex: "\\infty" },
  { sym: "→", tex: "\\to" }, { sym: "∈", tex: "\\in" },
  { sym: "∩", tex: "\\cap" }, { sym: "∪", tex: "\\cup" },
];

function insertIntoLatex(tex, caret) {
  const el = $("latexInput");
  const start = el.selectionStart ?? el.value.length;
  const end = el.selectionEnd ?? el.value.length;
  el.value = el.value.slice(0, start) + tex + el.value.slice(end);
  const pos = start + (caret ?? tex.length);
  el.focus();
  el.setSelectionRange(pos, pos);
}

const symbolPalette = $("symbolPalette");
for (const { sym, tex, caret } of SYMBOLS) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = sym;
  btn.title = tex;
  btn.addEventListener("click", () => insertIntoLatex(tex, caret));
  symbolPalette.appendChild(btn);
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
    // rendered=false: mathtext não entendeu o LaTeX e o carimbo seria texto
    // bruto, não uma fórmula -- NÃO troca pra ferramenta Carimbo sozinho,
    // senão o próximo clique no quadro posiciona esse lixo sem o usuário
    // ter decidido isso. Se ele realmente quiser carimbar mesmo assim,
    // escolhe a ferramenta Carimbo manualmente (gesto explícito).
    if (r.rendered) {
      showError($("latexError"), "");
      setTool("stamp"); // fantasma segue o cursor; clique posiciona
    } else {
      showError($("latexError"),
        "⚠ isso não é LaTeX válido — o renderizador não entendeu como fórmula (mostraria texto " +
        "bruto). Corrija o LaTeX e renderize de novo; se quiser carimbar o texto bruto mesmo " +
        "assim, escolha a ferramenta Carimbo manualmente.");
    }
  } catch (err) {
    showError($("latexError"), err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Renderizar";
  }
}
$("latexBtn").addEventListener("click", doRenderLatex);
$("latexInput").addEventListener("keydown", (e) => { if (e.key === "Enter") doRenderLatex(); });

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
