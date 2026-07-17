// Parser de expressões y = f(x) — descida recursiva, puro, sem DOM.
//
// Gramática (precedência crescente):
//   expr   := term (('+'|'-') term)*
//   term   := unary (('*'|'/') unary | IMPLÍCITA unary)*   // 2x, 3sin(x), (x+1)(x-2)
//   unary  := ('-'|'+') unary | power
//   power  := atom ('^' unary)?                            // assoc. à direita: 2^3^2 = 512
//   atom   := NUMBER | 'x' | 'pi' | 'e' | FUNC '(' expr ')' | '(' expr ')'
//
// Em vez de montar uma AST e depois interpretá-la, cada função de parse
// devolve diretamente um closure (x) => number — o "compilador" e o
// "avaliador" são a mesma passada.

export class ExprError extends Error {
  constructor(message, pos) {
    super(message);
    this.pos = pos;
  }
}

const FUNCS = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan,
  exp: Math.exp, log: Math.log, ln: Math.log,
  sqrt: Math.sqrt, abs: Math.abs,
};
const CONSTS = { pi: Math.PI, e: Math.E };

// palavras conhecidas, da mais longa para a mais curta, para o tokenizer
// casar "exp" antes de "e" e permitir coisas como "xsin(x)" => x*sin(x)
const WORDS = [...Object.keys(FUNCS), ...Object.keys(CONSTS), "x"]
  .sort((a, b) => b.length - a.length);

export function tokenize(src) {
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
      const w = WORDS.find((w) => src.startsWith(w, i));
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

export function parse(src) {
  const toks = tokenize(src);
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
      f = op === "+" ? (x) => h(x) + g(x) : (x) => h(x) - g(x);
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
        f = t.type === "*" ? (x) => h(x) * g(x) : (x) => h(x) / g(x);
      } else if (startsAtom(t)) {
        // multiplicação implícita: 2x, 3sin(x), (x+1)(x-2)
        const g = parseUnary(), h = f;
        f = (x) => h(x) * g(x);
      } else break;
    }
    return f;
  }

  function parseUnary() {
    const t = peek();
    if (t.type === "-") { next(); const g = parseUnary(); return (x) => -g(x); }
    if (t.type === "+") { next(); return parseUnary(); }
    return parsePower();
  }

  function parsePower() {
    const base = parseAtom();
    if (peek().type === "^") {
      next();
      const ex = parseUnary(); // unary, não power: é isso que dá assoc. à direita
      return (x) => Math.pow(base(x), ex(x));
    }
    return base;
  }

  function parseAtom() {
    const t = next();
    if (t.type === "num") { const v = t.value; return () => v; }
    if (t.type === "(") { const g = parseExpr(); expect(")"); return g; }
    if (t.type === "word") {
      if (t.value === "x") return (x) => x;
      if (t.value in CONSTS) { const v = CONSTS[t.value]; return () => v; }
      const fn = FUNCS[t.value];
      expect("(");
      const g = parseExpr();
      expect(")");
      return (x) => fn(g(x));
    }
    throw new ExprError("expressão incompleta", t.pos);
  }

  const f = parseExpr();
  expect("end");
  return f;
}

// API principal: aceita "y = ..." opcional, é case-insensitive.
export function compile(src) {
  let s = String(src).trim().toLowerCase();
  s = s.replace(/^y\s*=\s*/, "");
  if (!s) throw new ExprError("expressão vazia", 0);
  return parse(s);
}
