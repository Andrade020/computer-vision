# MathBoard — um paint matemático (m-paint)

Um "Paint" onde as ferramentas são matemáticas: além da caneta comum, você
desenha **só na horizontal/vertical** (modo `+`), transforma retas em
**pontes brownianas**, **plota curvas** — `y = f(x)`, **paramétricas
x(t), y(t)** e **polares r(t)** — sobre uma grade estilo GeoGebra,
**carimba fórmulas LaTeX** renderizadas no quadro e, com a ferramenta 🔍,
**desenha uma fórmula à mão e ela vira LaTeX** (OCR com pix2tex), pronta
para carimbar bonita no lugar dos rabiscos. O quadro é uma **folha
infinita**: pan e zoom movem o desenho inteiro.

Fatias entregues: **M1** (núcleo), **M2** (OCR), **M3** (folha infinita +
paramétricas/polares). Veja `/demo.html` para a mesma cena renderizada em
dois viewports.

## Rodando

```bash
cd m-paint
pip install -r requirements.txt
uvicorn server.app:app --reload --port 8000
# abra http://127.0.0.1:8000
```

Testes:

```bash
python -m pytest tests -q        # backend (FastAPI TestClient)
# frontend: abra http://127.0.0.1:8000/tests.html (tabela verde/vermelha)
```

## Arquitetura

```
server/                          # FastAPI
  app.py                         #   POST /api/render_latex, POST /api/ocr + estáticos
  mathimg.py                     #   LaTeX -> PNG via matplotlib mathtext
                                 #   (vendorizado de handwritten_text/hw/mathimg.py)
  ocr.py                         #   traços -> LaTeX via pix2tex (carga preguiçosa)
static/js/                       # vanilla ES modules — sem build, sem framework
  board.js                       #   3 canvases empilhados + eventos de ponteiro
  scene.js                       #   cena vetorial + undo/redo por comandos
  viewport.js                    #   matemática <-> pixels, grade, ticks 1-2-5
  expr.js                        #   parser de expressões (descida recursiva)
  plot.js                        #   amostragem de y=f(x) -> stroke
  rng.js                         #   mulberry32 + Box-Muller
  api.js                         #   cliente do backend
  tools/{pen,manhattan,brownian,stamp,ocr}.js
```

O backend só faz o que o navegador não faz sozinho: rasterizar LaTeX
(matplotlib mathtext, offline, sem instalação de TeX). Todo o resto — desenho,
desfazer, parser, grade — é JavaScript puro em módulos ES, sem nenhuma
dependência de frontend.

### As três camadas de canvas

| camada    | conteúdo                          | quando redesenha            |
|-----------|-----------------------------------|-----------------------------|
| `grid`    | grade, eixos, labels              | pan/zoom/toggle             |
| `ink`     | tudo que foi commitado            | replay a cada mutação       |
| `preview` | traço ao vivo, fantasma do stamp  | limpa a cada frame          |

### Undo por replay, não por snapshot

A cena é uma lista de objetos vetoriais (`stroke` com polylines, `stamp` com
imagem, `curve` com expressão, `wipe` com região). Desfazer = remover o
último objeto e **redesenhar tudo**. Um snapshot RGBA de 1600×1000 custaria
~6,4 MB por passo de histórico; redesenhar algumas centenas de polylines
custa menos de 10 ms. Bônus: redimensionar a janela não perde nada, e
`Limpar` também é desfazível (a operação guarda a lista removida). A
borracha é um stroke com `globalCompositeOperation: 'destination-out'` —
como o replay preserva a ordem dos acontecimentos, ela "fura" exatamente o
que existia quando você apagou.

### Folha infinita: a cena vive em coordenadas matemáticas (M3)

As ferramentas trabalham em pixels (é onde o mouse mora), mas na hora do
commit o Board converte o objeto com `toMathObject()` e a cena guarda tudo
em **coordenadas matemáticas** — inclusive a largura do traço, em unidades.
No replay, `drawObject()` transforma de volta para o viewport atual. As
consequências caem de graça:

- **pan e zoom movem o desenho inteiro** (tinta, carimbos, regiões apagadas),
  como arrastar uma folha sob uma lupa;
- dar zoom **amplia o traço** junto — é o modelo mental de lupa sobre papel;
- as **curvas são a exceção deliberada**: guardam a expressão, não os pontos,
  e são **re-amostradas a cada repaint** para o viewport corrente. Por isso
  nunca ficam poligonais ao ampliar (comportamento GeoGebra) e mantêm largura
  constante em pixels, como objetos matemáticos que são.

### Curvas: cartesianas, paramétricas e polares

Três formas no mesmo painel: `y = f(x)` (1 amostra a cada ~2 px de coluna),
`x(t), y(t)` e `r(t)` (1200 amostras em `t ∈ [t0, t1]`; os campos de faixa
aceitam expressões constantes como `2pi`). Amostras não finitas, pontos
muito além do viewport ou saltos grandes em pixels quebram a polyline em
ramos — `tan(x)` vira uma família de ramos sem espigões verticais, e uma
rosácea `r = 2cos(3t)` fecha perfeitamente.

### Modo `+` (Manhattan)

Uma máquina de estados com **histerese**: nenhum eixo é escolhido até o cursor
sair de uma zona morta de 6 px (aí o eixo dominante vence); o cursor é
projetado no trilho atual; e só quando o desvio perpendicular passa de 12 px
um canto é commitado e o eixo vira. Sem a histerese, um arraste quase diagonal
trocaria de eixo a cada pixel e viraria uma escadinha trêmula.

### Ponte browniana

Você arrasta uma reta A→B e ela vira um passeio aleatório **condicionado a
terminar exatamente em B**. Construção clássica: dado o passeio
`S_k = Σ g_i·σ·√Δt` com `g_i ~ N(0,1)`,

```
bridge_k = S_k − (k/N)·S_N
```

zera as duas pontas. O desvio é aplicado na direção *normal* ao segmento, e a
escala `σ·√L` torna o slider de volatilidade invariante ao comprimento do
traço. Detalhe de interação: a seed do gerador (`mulberry32`) é sorteada no
`pointerdown` e **fixada durante o arraste** — o preview estica elasticamente
em vez de "ferver" com ruído novo a cada movimento. σ = 0 devolve a reta
exata (e é um dos testes).

### Parser de expressões

Descida recursiva de ~150 linhas, zero dependências (math.js seria ~700 KB
para esta gramática):

```
expr  := term (('+'|'-') term)*
term  := unary (('*'|'/') unary | IMPLÍCITA unary)*   # 2x, 3sin(x), (x+1)(x-2)
unary := ('-'|'+') unary | power
power := atom ('^' unary)?                            # à direita: 2^3^2 = 512
atom  := NUMBER | VAR | CONST | FUNC '(' expr ')' | '(' expr ')'
VAR   := x (cartesiana) | t (paramétrica/polar)       # configurável
CONST := pi tau e
FUNC  := sin cos tan asin acos atan sinh cosh tanh
         exp log ln sqrt abs floor ceil round sign
```

Em vez de montar uma AST e interpretá-la depois, cada função de parse devolve
diretamente um closure `(v) => number` — parser e compilador na mesma passada.
Erros carregam a posição do caractere e aparecem inline na UI. O tokenizer
casa palavras da mais longa para a mais curta, então `tan` nunca é engolido
pela variável `t`, e `xsin(x)` vira `x*sin(x)`.

### Carimbo LaTeX

`POST /api/render_latex {latex, height_px, color}` → PNG transparente em
base64 + dimensões. O mathtext do matplotlib cobre a maior parte do LaTeX
matemático sem nenhum TeX instalado; se a expressão não parseia, cai para
texto literal (`\mathrm`) em vez de dar erro. O frontend pede o dobro da
altura de exibição e mostra na metade — nítido em telas hiDPI.

### OCR de fórmula desenhada (M2)

O fluxo completo: com a ferramenta 🔍 você arrasta um retângulo em volta da
fórmula desenhada; o frontend recorta a camada de tinta (na resolução do
backing store) e manda para `POST /api/ocr`; o **pix2tex** (LaTeX-OCR, um
ViT de ~100 MB rodando em CPU) devolve o LaTeX; o texto cai no campo de
fórmula, é renderizado pelo caminho já existente do carimbo, e — se a opção
"apagar traços reconhecidos" estiver marcada — os rabiscos originais somem
(um objeto `wipe` na cena, desfazível como tudo).

Detalhes que fazem diferença:

- **Carga preguiçosa**: o modelo só é carregado na primeira requisição
  (e baixado na primeira execução da máquina); subir o servidor é instantâneo.
- **`prepare_for_ocr`**: pix2tex foi treinado em fórmulas *impressas*, então
  o recorte transparente e colorido do canvas é fundido sobre branco,
  `autocontrast` estica a tinta mais escura até o preto (funciona para
  qualquer cor de caneta), recorta no bounding box e ganha margem branca.
- **Limpeza pós-OCR**: o modelo adora prefixar `\scriptstyle{...}` em
  entradas pequenas; `cleanup_latex` remove estilos e desembrulha chaves
  externas antes de mostrar.
- **Aviso de baixa confiança**: o backend conta os "componentes de tinta"
  significativos da seleção (`server/ocr.py::ink_complexity`, via
  `scipy.ndimage.label`, filtrando manchas de anti-aliasing). Seleções com
  4 componentes ou menos — na prática, uma letra, um dígito, "2x" — disparam
  `low_confidence: true` na resposta; a UI então segura o aviso na tela
  (em vez de escondê-lo depois de alguns segundos como no caso normal) e
  **não apaga os traços originais**, mesmo com a opção marcada, porque
  apagar um rabisco para colocar lixo no lugar seria perder trabalho à toa.
  Ver a seção de limitações abaixo para o porquê disso ser necessário.

## Limitações honestas

- O parser conhece uma variável por expressão e funções de um argumento só
  (sem `min(a,b)`, sem `atan2`); nada de variáveis definidas pelo usuário.
- Sem persistência: F5 limpa o quadro (salvar/carregar a cena como JSON é a
  candidata natural para a M4 — a cena já é uma lista de objetos simples).
- Não dá para selecionar/mover objetos já desenhados; edição é desenhar,
  apagar e desfazer.
- mathtext ≠ TeX completo: ambientes como `\begin{align}` não existem.
- **O OCR é ruim em expressões curtas — e isso não é um bug de
  pré-processamento, é o modelo mesmo.** pix2tex foi treinado em fórmulas
  extraídas de artigos científicos (im2latex-100k), onde uma expressão
  isolada como "2x" ou "x" sozinho praticamente não aparece como fórmula
  completa. Sem calibração para "isto é trivial", o modelo alucina algo
  visualmente parecido com LaTeX complexo. Testado sistematicamente:
  `2` → `\stackrel{\prime\prime}{\bigcup}`, `x` → `\mathcal{N}`,
  `2x` → `{\mathcal{D}}X`, `x+1` → `X+1` (letra errada) — todos lixo ou
  quase. Em compensação, fórmulas com mais estrutura funcionam bem de
  verdade: `x^2+2x+1`, `\frac{a+b}{c}`, `\int_0^1 x\,dx`,
  `\sum_{k=1}^n k = \frac{n(n+1)}{2}` voltaram perfeitas, inclusive com
  distorção elástica simulando traço trêmulo. Tentei mitigar reescalando
  seleções pequenas antes do OCR; a mudança ajudava alguns casos triviais
  mas **piorava** fórmulas que já funcionavam (`\frac{x^2+1}{2}` virava
  `x^{\frac{x^2+1}{2}}`) — não é um trade-off que vale a pena, então não
  entrou no código. Na prática: para algo curto, digitar direto no campo
  LaTeX é mais rápido e confiável do que brigar com o OCR.
- A primeira requisição de OCR é lenta (carga do modelo, ~10 s em CPU);
  as seguintes levam ~1–3 s.

## Roadmap

- **M4 (ideias)** — salvar/carregar a cena (JSON; carimbos re-renderizam a
  partir do LaTeX guardado), selecionar/mover objetos, funções de dois
  argumentos no parser.
