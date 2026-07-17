// PRNG com seed + gaussianas — puro, sem DOM.
//
// mulberry32: gerador de 32 bits minúsculo e bom o suficiente para gráficos.
// A seed fixa é o que permite o preview "elástico" da ponte browniana:
// a cada pointermove a ponte inteira é regenerada com a MESMA seed, então o
// ruído não "ferve" enquanto o usuário arrasta.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller: transforma pares uniformes em pares N(0,1).
// Devolve uma função que entrega uma gaussiana por chamada (guarda a sobra).
export function gaussians(rng) {
  let spare = null;
  return function () {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    do { u = rng(); } while (u === 0); // log(0) = -inf
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * v;
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

export function randomSeed() {
  return (Math.random() * 2 ** 32) >>> 0;
}
