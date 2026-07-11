# Neural Handwriting — escreve qualquer coisa (e LaTeX) com a sua letra

Transforma o projeto original (transcrever texto em imagem manuscrita) numa
**pipeline neural de síntese de escrita à mão**: dá um texto ou um documento
LaTeX e recebe uma página que parece escrita à mão pela sua letra.

## Por que esta arquitetura (dado o hardware)

A máquina é **CPU-only** (i7-1355U, sem GPU NVIDIA — só Intel Iris Xe), 32 GB RAM.
Os dados são ~2.200 glifos isolados da sua letra real, segmentados por OCR,
muito desbalanceados e ruidosos. **Não há dados de trajetória de caneta**, então
síntese sequencial (estilo Graves RNN) não é viável aqui.

A escolha que dá o melhor resultado nesse cenário é **híbrida**:

1. **Banco de tinta real** — cada caractere é renderizado usando os seus traços
   reais. Como um exemplo diferente é sorteado a cada ocorrência, letras
   repetidas saem diferentes → cara de manuscrito de verdade. É a fonte de maior
   fidelidade e funciona na hora, sem treino.
2. **Rede neural (CVAE condicional)** — gera qualquer caractere a partir de um
   vetor de estilo `z ~ N(0, I)`. Serve para (a) preencher caracteres com poucos
   ou nenhum exemplo, (b) variação infinita, (c) o requisito "rede que escreve
   qualquer coisa". Treina em CPU durante a noite.
3. **Renderizador** — motor de fluxo que posiciona glifos com baseline, ascendentes/
   descendentes, inclinação, jitter e quebra de linha; mistura glifos manuscritos
   e imagens (matemática) na mesma linha.
4. **Front-end LaTeX** — prosa vira manuscrito; matemática (`$...$`, `\[...\]`,
   `equation`, `align`) é tipografada (matplotlib mathtext, offline) e encaixada
   inline/centralizada. Isso é "compilar LaTeX com a sua letra".

## Estrutura

```
hw/
  data_build.py    limpa/normaliza os glifos -> data/train.npz + data/glyph_bank.pkl
  metrics.py       tabela tipográfica (ascendentes/descendentes/caixa por char)
  model.py         ConditionalVAE + GlyphGenerator (inferência)
  train.py         treino em CPU, checkpoints + grades de amostra, resumível
  mathimg.py       LaTeX math -> imagem (mathtext; MiKTeX opcional)
  render.py        HandwritingRenderer: texto/documento -> página(s)
  latex_render.py  parser de um subconjunto comum de LaTeX -> blocos
handwrite.py         CLI: texto -> PNG manuscrito
handwrite_latex.py   CLI: .tex -> PNG por página + PDF
```

## Uso

```bash
# 1) (re)construir o dataset limpo + banco de glifos
python hw/data_build.py

# 2) texto simples -> manuscrito
python handwrite.py "Qualquer texto na minha letra." -o out/nota.png --ruled

# 3) LaTeX -> manuscrito (prosa na sua letra, matemática tipografada)
python handwrite_latex.py out/sample.tex -o out/doc --ruled

# 4) usar a rede neural como fallback para caracteres raros/ausentes
python handwrite.py "..." --model
```

## Treino da rede

```bash
python -m hw.train --epochs 6000 --save_every 100      # começar
python -m hw.train --epochs 6000 --resume              # retomar de last.pt
```

- ~5–10 s por época em CPU. Checkpoints: `hw/checkpoints/{last,best}.pt`.
- Grades de amostra por época em `hw/checkpoints/../samples/epoch_*.png` para
  acompanhar o aprendizado.
- Log CSV: `hw/checkpoints/train_log.csv` (loss/bce/kld/beta/tempo).

## Limitações honestas

- Os rótulos vindos do OCR têm ruído; o `data_build.py` remove blocos preenchidos
  e outliers de densidade, mas alguns glifos imperfeitos podem sobrar.
- Faltam exemplos de vários caracteres (dígitos, maiúsculas raras, `k w y z v s`
  minúsculos). O renderizador usa fallback de caixa (maiúscula↔minúscula) e a
  rede; para qualidade máxima nesses, o ideal é coletar mais amostras da sua letra.
- A matemática é tipografada (não manuscrita), porque os símbolos matemáticos não
  existem no seu banco de glifos — é a decisão que mantém as equações corretas.
- É um subconjunto de LaTeX (notas/listas/seções/matemática), não um engine TeX.

## Próximo passo de maior impacto na qualidade

Coletar um conjunto **limpo e completo** da sua letra: uma folha com o alfabeto
(minúsculas + maiúsculas), dígitos e pontuação, algumas repetições cada. Isso
melhora o banco e o treino mais do que qualquer ajuste de modelo.
