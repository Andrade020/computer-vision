# Neural Handwriting — escreve qualquer coisa (e LaTeX) com a sua letra

Transforma o projeto original (transcrever texto em imagem manuscrita) numa
**pipeline neural de síntese de escrita à mão**: dá um texto ou um documento
LaTeX/Markdown e recebe uma página que parece escrita à mão pela sua letra —
inclusive a **matemática** (integrais, somatórios, raízes, frações, gregas)
com os seus glifos reais e os símbolos desenhados na mesma tinta. Também
monta **documentos de verdade**: título e número de página em cada folha,
tabelas e figuras manuscritas embutidas no meio do texto (ver "Motor de
documento" abaixo).

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

## Motor de documento: título, paginação, tabelas e figuras

Até aqui o motor produzia *texto* manuscrito; esta fatia acrescenta a
estrutura de um *documento* de verdade, sem sair do mesmo pipeline de glifos:

- **Título e numeração de página** (`--title`, numeração ligada por padrão em
  `handwrite_latex.py`/`handwrite_markdown.py`/na GUI) são carimbados
  diretamente na página já pronta, usando os MESMOS glifos manuscritos do
  corpo do texto (`HandwritingRenderer.stamp_header_footer`) — não é uma
  fonte de sistema jogada em cima, o que quebraria a ilusão de manuscrito.
  Isso mantém a arquitetura "lazy"/streaming intacta: o carimbo acontece
  página a página, no mesmo loop que já salva cada PNG, sem exigir dois
  passes pelo documento nem saber o total de páginas de antemão (por isso
  é "Página 3", não "Página 3 de 10" — ver "Limitações honestas").
- **Tabelas**: sintaxe padrão de tabela do Markdown (`| a | b |` + linha
  separadora `|---|---|`) vira um bloco `table`, desenhado como uma grade de
  verdade (linhas finas) com cada célula em letra manuscrita.
- **Figuras**: sintaxe padrão `![legenda](caminho/da/imagem.png)` vira um
  bloco `figure` — a imagem é redimensionada para caber na largura
  disponível (sem nunca aumentar além do tamanho original) e a legenda,
  se houver, é desenhada centralizada embaixo em letra manuscrita.
  Caminhos relativos são resolvidos a partir da pasta do `.md` de entrada,
  não do diretório onde você roda o comando.

## Estrutura

```
hw/
  data_build.py    limpa/normaliza os glifos -> data/train.npz + data/glyph_bank.pkl
  metrics.py       tabela tipográfica (ascendentes/descendentes/caixa por char)
  imageops.py      ops leves (numpy+scipy): normalizacao de espessura de traco,
                   textura de densidade de tinta, tremor elastico (compartilhado
                   com mathhand.py)
  model.py         ConditionalVAE + GlyphGenerator (inferência)
  train.py         treino em CPU, checkpoints + grades de amostra, resumível
  mathimg.py       LaTeX math -> imagem tipografada (mathtext; fallback)
  mathhand.py      motor de layout matematico (mini-TeX): seus glifos reais +
                   simbolos (int, sum, sqrt, gregas) na sua tinta -> --hand-math
  render.py        HandwritingRenderer: texto/documento -> página(s), com
                   iter_document() gerando pagina a pagina (lazy/streaming);
                   blocos "figure"/"table" (imagem+legenda / grade de
                   celulas); stamp_header_footer() carimba titulo/numero
                   de pagina numa pagina ja pronta, em letra manuscrita
  latex_render.py  parser de um subconjunto comum de LaTeX -> blocos
  markdown_render.py parser de Markdown+LaTeX (headers, **negrito**, ---,
                   listas, $...$/$$...$$, tabelas `|a|b|`, figuras
                   `![legenda](caminho)`) -> blocos
  paper.py         efeito de papel escaneado (warp, dobras/sombras, grao,
                   variacao de luz) -> --scan nos CLIs
  keep_awake.py    impede o sono do Windows durante treinos longos (reversível)
handwrite.py         CLI: texto -> PNG manuscrito
handwrite_latex.py   CLI: .tex -> PNG por página + PDF
handwrite_markdown.py CLI: .md (Markdown+LaTeX) -> PNG por página + PDF,
                     salvando cada pagina no disco assim que fica pronta
handwrite_gui.py     interface Tkinter: digite ou importe um .md/.tex/.txt,
                     ajuste as mesmas opcoes dos CLIs, gere com preview ao vivo
finalize.py          gera o showcase (alfabeto da rede, demos, PDF) + stats
```

## Interface gráfica

```bash
python handwrite_gui.py
```

Janela com visual próprio (customtkinter — cartões arredondados, switches,
sliders, tema claro papel/tinta com a logo do projeto), não o Tkinter padrão
"cru". Escolha o modo de conteúdo (Markdown+LaTeX / LaTeX puro / texto
simples), digite direto no editor ou clique "Importar arquivo..." para
carregar um `.md`/`.tex`/`.txt`. As mesmas opções dos CLIs ficam disponíveis
em cartões (Opções / Ajustes finos): matemática manuscrita, papel escaneado,
tinta, tremor, título do documento (opcional) e numeração de páginas (ligada
por padrão), etc. A geração roda em uma thread separada (a janela não
trava) usando o mesmo `iter_document` lazy dos CLIs — cada página é salva no
disco assim que fica pronta e aparece como preview ao vivo num cartão da
janela, com barra de progresso bloco a bloco. Ao final, monta o PDF.

Requer `customtkinter` (`pip install customtkinter`) além de `tkinter` (já
vem com o Python padrão no Windows). O pipeline novo (`hw/`) não usa `cv2`
em lugar nenhum — só o `app.py` antigo (Tkinter puro, não usado por este
README) depende dele.

Ícone/logo: `assets/icon.ico` (multi-resolução, usado na barra de título e na
taskbar) e `assets/logo.png` (recorte com fundo transparente, mostrado no
cabeçalho da janela), gerados a partir da arte original do projeto.

## Uso (linha de comando)

```bash
# 1) (re)construir o dataset limpo + banco de glifos
python hw/data_build.py

# 2) texto simples -> manuscrito
python handwrite.py "Qualquer texto na minha letra." -o out/nota.png --ruled
#     --regularize 1 uniformiza a espessura dos tracos (corrige o corte desigual);
#     --stroke 0.11 define a espessura-alvo (fracao da altura-x); 0=desliga

# 3) LaTeX -> manuscrito (prosa na sua letra, matemática tipografada)
python handwrite_latex.py out/sample.tex -o out/doc --ruled

# 3b) LaTeX com a MATEMATICA tambem na sua letra (integrais, somatorios, raizes...)
python handwrite_latex.py out/math_showcase.tex -o out/math --ruled --hand-math
#     --math-style controla o "tremido" dos simbolos: 0=vetorial limpo,
#     1=padrao, 1.5+=mais rustico (ex.: --math-style 1.5)

# 4) usar a rede neural como fallback para caracteres raros/ausentes
python handwrite.py "..." --model

# 5) documento Markdown+LaTeX longo (ex.: resolucao de lista com dezenas de
#    paginas e centenas de equacoes) -> PNG por pagina + PDF, salvando cada
#    pagina no disco assim que fica pronta (nao acumula tudo em memoria) e
#    logando o progresso -- documentos de 400+ linhas / ~200 blocos / 450+
#    expressoes matematicas renderizam em poucos segundos
python handwrite_markdown.py resolucao.md -o out/resolucao --ruled --hand-math

# 6) efeito de papel escaneado (warp leve, dobras/sombras, grao, luz irregular)
#    em qualquer um dos CLIs acima: --scan (=1.0) ou --scan 1.5 (mais forte)
python handwrite.py "..." -o out/nota.png --ruled --scan

# 7) tinta e flutuacao das letras (ligados por padrao em todos os CLIs):
#    --ink 0.8    textura de densidade de tinta dentro do traco (0=chapado)
#    --tremor 0.3 leve ondulacao na forma da letra real (0=forma crua do banco)
python handwrite.py "..." -o out/nota.png --ink 1.0 --tremor 0.5

# 8) titulo + numeracao de pagina (numeracao ligada por padrao em
#    handwrite_latex.py/handwrite_markdown.py -- use --no-page-numbers p/ desligar)
python handwrite_markdown.py resolucao.md -o out/resolucao --title "Lista 3 - Econometria"
python handwrite_latex.py doc.tex -o out/doc --title "Meu Documento" --no-page-numbers

# 9) tabelas e figuras dentro de um .md (sintaxe padrao, sem flag extra):
#    | Coluna A | Coluna B |
#    | --- | --- |
#    | valor 1  | valor 2  |
#
#    ![legenda opcional](figs/grafico.png)
python handwrite_markdown.py doc_com_tabela_e_figura.md -o out/doc
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
- Espessura de traço: como o recorte da segmentação deixou cada glifo numa
  resolução diferente, a espessura saía desigual. `--regularize` normaliza a
  largura de traço de cada glifo no tamanho final (dilata finos / afina grossos)
  para um alvo comum (`--stroke`), no texto e na matemática. Os símbolos usam o
  mesmo alvo, então casam de peso com as suas letras.
- Faltam exemplos de vários caracteres (dígitos, maiúsculas raras, `k w y z v s`
  minúsculos). O renderizador usa fallback de caixa (maiúscula↔minúscula) e a
  rede; para qualidade máxima nesses, o ideal é coletar mais amostras da sua letra.
- Matemática: por padrão é tipografada; com `--hand-math`, as letras e dígitos da
  equação usam seus glifos reais e os símbolos estruturais (∫ ∑ √ ∏ frações,
  expoentes/índices, gregas, operadores) são desenhados na mesma tinta. Símbolos
  que você nunca escreveu não têm como sair na sua mão exata — saem no estilo que
  combina. Para não ficarem "perfeitos demais", cada símbolo é tratado como uma
  forma-*prior* (mathtext) e re-estilizado com características medidas da sua letra
  (espessura de traço estimada do banco, tremor de baixa frequência, rugosidade de
  borda e variação de tinta), reamostrado a cada render. A intensidade é o knob
  `--math-style` (0=limpo, 1=padrão, 1.5+=mais rústico). `hw/mathhand.py` cobre um
  subconjunto comum (int/sum/prod/lim, frac, sqrt, ^/_, \left..\right, gregas, ops).
- É um subconjunto de LaTeX (notas/listas/seções/matemática), não um engine TeX.
- Matrizes (`bmatrix`/`pmatrix`/`vmatrix`/`matrix`), `cases` e acentos
  (`\hat \bar \tilde \vec \dot`) são suportados por `mathhand.py`, com
  `\mathbb`/`\mathcal`/`\boldsymbol` renderizados como o conteúdo interno (sem
  a fonte especial, que não existe na sua letra).
- Pontuação comum (`. , : ; - !`) que não está no banco (a segmentação OCR só
  capturou letras/dígitos) é desenhada proceduralmente e passa pelo mesmo
  pipeline de regularização/jitter dos glifos reais, em vez de sumir da página.
- Documentos longos: `iter_document()`/`handwrite_markdown.py` geram e salvam
  cada página assim que fica pronta (não acumulam o documento inteiro em
  memória) e cada bloco roda isolado em try/except — uma construção malformada
  é pulada (e reportada) em vez de derrubar o restante do documento. Um cache
  do raster do mathtext (antes da distorção manuscrita, que continua variando
  a cada render) evita reprocessar símbolos repetidos centenas de vezes.

- Papel escaneado (`--scan`): o projeto original (`writting_colos.py`,
  `simulate_paper_folds`/`simulate_ink`) já fazia isso com `cv2`, que não está
  instalado nesta máquina — por isso o pipeline neural nunca teve essa etapa.
  `hw/paper.py` reimplementa os mesmos cinco ingredientes só com PIL+numpy+scipy
  (sem cv2): warp senoidal leve, sombras de dobra aleatórias, ruído de baixa
  frequência, gradiente vertical de brilho e grão fino. `--scan` liga o efeito
  (padrão 1.0); `--scan 1.5` deixa mais gasto/dobrado, `--scan 0.5` mais sutil.

- Tinta e flutuação das letras (`--ink`, `--tremor`): a segmentação OCR
  binariza o glifo com um threshold duro (`fg = a > 127` em `data_build.py`),
  então a densidade de tinta real da digitalização original foi descartada — as
  letras saíam com tinta 100% sólida e uniforme, sem a variação de pressão de
  uma caneta de verdade. `hw/imageops.py::ink_texture` reintroduz isso de forma
  sintética (núcleo do traço quase opaco, com manchas ocasionais mais claras
  via ruído de baixa frequência) e `elastic()` (compartilhada com o motor de
  matemática) dá um leve tremor de forma. Ambos rodam por padrão (`--ink 0.8`,
  `--tremor 0.3`); `--ink 0` volta ao chapado antigo, `--tremor 0` mantém a
  forma crua do banco (a variedade real já vem de cada ocorrência sortear uma
  amostra diferente do banco + o jitter de rotação/posição que já existia).

- **Numeração "Página X", não "Página X de N"**: `iter_document()` é
  deliberadamente lazy/streaming (gera e salva cada página sem nunca
  carregar o documento inteiro em memória, essencial para documentos
  longos). Saber o total de páginas de antemão exigiria processar o
  documento inteiro duas vezes (uma só para contar, outra para renderizar
  com o total certo) — decidido que não valia o custo para esta fatia;
  "Página X" sequencial já é o suficiente para a maioria dos cadernos/notas.
- **Tabelas são uma grade simples**: colunas de largura igual, uma linha por
  célula (texto mais longo que a coluna é cortado, não quebra em várias
  linhas dentro da célula), sem suporte a `**negrito**`/`$math$` dentro da
  célula (o conteúdo é tratado como texto puro). Uma tabela maior que o
  espaço restante da página inteira começa numa página nova — não é
  dividida no meio (nenhuma linha de tabela é cortada ao meio).
- **Figuras nunca ampliam além do tamanho original** (só encolhem para caber
  na largura/altura disponível), mesma convenção do `resize_image` do
  projeto irmão `classical_filters`.
- **Sumário (TOC) e notas de rodapé ainda não existem** — são a próxima
  fatia planejada deste motor de documento (rastrear títulos/página para
  montar um sumário, e âncoras de nota de rodapé com o texto no rodapé da
  mesma página). Por enquanto, `#`/`##` só controlam o tamanho da letra do
  título, sem entrar em nenhum índice.

## Próximo passo de maior impacto na qualidade

Coletar um conjunto **limpo e completo** da sua letra: uma folha com o alfabeto
(minúsculas + maiúsculas), dígitos e pontuação, algumas repetições cada. Isso
melhora o banco e o treino mais do que qualquer ajuste de modelo.
