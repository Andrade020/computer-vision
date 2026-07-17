// Parser de expressões — descida recursiva, puro, sem DOM.
//
// Gramática (precedência crescente):
//   expr   := term (('+'|'-') term)*
//   term   := unary (('*'|'/') unary | IMPLÍCITA unary)*   // 2x, 3sin(x), (x+1)(x-2)
//   unary  := ('-'|'+') unary | power
//   power  := atom ('^' unary)?                            // assoc. à direita: 2^3^2 = 512
//   atom   := NUMBER | VAR | CONST | FUNC '(' expr ')' | '(' expr ')'
//
// A variável é configurável: 'x' para y=f(x), 't' para paramétricas/polares.
// Em vez de montar uma AST e depois interpretá-la, cada função de parse
// devolve diretamente um closure (v) => number — o "compilador" e o
// "avaliador" são a mesma passada.

export class ExprError extends Error {
  constructor(message, pos) {
    super(message);
    this.pos = pos;
  }
}

const FUNCS = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan,
  asin: Math.asin, acos: Math.acos, atan: Math.atan,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
  exp: Math.exp, log: Math.log, ln: Math.log,
  sqrt: Math.sqrt, abs: Math.abs,
  floor: Math.floor, ceil: Math.ceil, round: Math.round, sign: Math.sign,
};
const CONSTS = { pi: Math.PI, tau: 2 * Math.PI, e: Math.E };

// palavras conhecidas, da mais longa para a mais curta, para o tokenizer
// casar "exp" antes de "e", "tan" antes de "t", e permitir "xsin(x)" => x*sin(x)
function wordsFor(variable) {
  return [...Object.keys(FUNCS), ...Object.keys(CONSTS), variable]
    .sort((a, b) => b.length - a.length);
}

export function tokenize(src, variable = "x") {
  const words = wordsFor(variable);
  const toks = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === " " || c === "\t") { i++; continue; }
    if (/[0-9.]/.test(c)) {
      const m = /^(?:\d+\.?\d*|\.\d+)/.exec(src.slice(i));
      if (!m) throw new ExprError("número inválido", i);
      toks.push({ type: "num", value: parseFloat(m[0]), pos: i });
      i += m[0].length;
      continue;
    }
    if (/[a-z]/.test(c)) {
      const w = words.find((w) => src.startsWith(w, i));
      if (!w) throw new ExprError(`símbolo desconhecido perto de "${src.slice(i, i + 6)}"`, i);
      toks.push({ type: "word", value: w, pos: i });
      i += w.length;
      continue;
    }
    if ("+-*/^()".includes(c)) {
      toks.push({ type: c, pos: i });
      i++;
      continue;
    }
    throw new ExprError(`caractere inesperado "${c}"`, i);
  }
  toks.push({ type: "end", pos: src.length });
  return toks;
}

export function parse(src, variable = "x") {
  const toks = tokenize(src, variable);
  let k = 0;
  const peek = () => toks[k];
  const next = () => toks[k++];
  const expect = (type) => {
    if (peek().type !== type) throw new ExprError(`esperado "${type}"`, peek().pos);
    return next();
  };
  const startsAtom = (t) => t.type === "num" || t.type === "word" || t.type === "(";

  function parseExpr() {
    let f = parseTerm();
    while (peek().type === "+" || peek().type === "-") {
      const op = next().type;
      const g = parseTerm(), h = f;
      f = op === "+" ? (v) => h(v) + g(v) : (v) => h(v) - g(v);
    }
    return f;
  }

  function parseTerm() {
    let f = parseUnary();
    for (;;) {
      const t = peek();
      if (t.type === "*" || t.type === "/") {
        next();
        const g = parseUnary(), h = f;
        f = t.type === "*" ? (v) => h(v) * g(v) : (v) => h(v) / g(v);
      } else if (startsAtom(t)) {
        // multiplicação implícita: 2x, 3sin(x), (x+1)(x-2)
        const g = parseUnary(), h = f;
        f = (v) => h(v) * g(v);
      } else break;
    }
    return f;
  }

  function parseUnary() {
    const t = peek();
    if (t.type === "-") { next(); const g = parseUnary(); return (v) => -g(v); }
    if (t.type === "+") { next(); return parseUnary(); }
    return parsePower();
  }

  function parsePower() {
    const base = parseAtom();
    if (peek().type === "^") {
      next();
      const ex = parseUnary(); // unary, não power: é isso que dá assoc. à direita
      return (v) => Math.pow(base(v), ex(v));
    }
    return base;
  }

  function parseAtom() {
    const t = next();
    if (t.type === "num") { const v = t.value; return () => v; }
    if (t.type === "(") { const g = parseExpr(); expect(")"); return g; }
    if (t.type === "word") {
      if (t.value === variable) return (v) => v;
      if (t.value in CONSTS) { const c = CONSTS[t.value]; return () => c; }
      const fn = FUNCS[t.value];
      expect("(");
      const g = parseExpr();
      expect(")");
      return (v) => fn(g(v));
    }
    throw new ExprError("expressão incompleta", t.pos);
  }

  const f = parseExpr();
  expect("end");
  return f;
}

// API principal. Aceita prefixos opcionais tipo "y =", "x(t) =", "r =";
// é case-insensitive.
export function compile(src, variable = "x") {
  let s = String(src).trim().toLowerCase();
  s = s.replace(/^[a-z]\s*(\(\s*[a-z]+\s*\))?\s*=\s*/, "");
  if (!s) throw new ExprError("expressão vazia", 0);
  return parse(s, variable);
}

// avalia uma expressão constante ("2pi", "pi/2", "-3") para campos de faixa;
// a variável-sentinela "\0" nunca casa com letra alguma, então "x" ou "t"
// soltos falham como símbolo desconhecido em vez de virarem 0 em silêncio
export function evalConst(src) {
  const v = compile(src, "\0")(0);
  if (!Number.isFinite(v)) throw new ExprError("valor não finito", 0);
  return v;
}
