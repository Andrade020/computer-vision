# MathBoard — um paint matemático (m-paint)

Um "Paint" onde as ferramentas são matemáticas: além da caneta comum, você
traça **linhas retas travadas em horizontal/vertical**, transforma retas em
**pontes brownianas** (com volatilidade ajustável), desenha círculos com um
**compasso de verdade** (fixa o centro, depois desenha o raio), **plota
curvas** — `y = f(x)`, **paramétricas x(t), y(t)** e **polares r(t)** — e
**carimba fórmulas LaTeX** renderizadas no quadro, com uma paleta de
símbolos matemáticos para montar a fórmula sem decorar comandos. O quadro é
uma **folha infinita**: pan e zoom movem o desenho inteiro.

Fatias entregues: **M1** (núcleo), **M2** (OCR — atualmente escondido da UI,
ver seção própria), **M3** (folha infinita + paramétricas/polares), mais uma
rodada de ajustes de usabilidade (linha, compasso, paleta de símbolos,
carimbo de uso único). Veja `/demo.html` para a mesma cena renderizada em
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
  ocr.py                         #   traços -> LaTeX via pix2tex (sem uso na UI, ver "OCR")
static/js/                       # vanilla ES modules — sem build, sem framework
  board.js                       #   3 canvases empilhados + eventos de ponteiro
  scene.js                       #   cena vetorial + undo/redo por comandos
  viewport.js                    #   matemática <-> pixels, grade, ticks 1-2-5
  expr.js                        #   parser de expressões (descida recursiva)
  plot.js                        #   amostragem de y=f(x) -> stroke
  rng.js                         #   mulberry32 + Box-Muller
  api.js                         #   cliente do backend
  tools/{pen,line,brownian,compass,stamp,hand}.js
                                 #   ocr.js também existe, sem uso (ver "OCR")
```

O backend só faz o que o navegador não faz sozinho: rasterizar LaTeX
(matplotlib mathtext, offline, sem instalação de TeX). Todo o resto — desenho,
desfazer, parser, grade — é JavaScript puro em módulos ES, sem nenhuma
dependência de frontend.

### As três camadas de canvas

| camada    | conteúdo                          | quando redesenha            |
|-----------|-----------------------------------|-----------------------------|
| `grid`    | grade, eixos, labels (off por padrão) | pan/zoom/toggle         |
| `ink`     | tudo que foi commitado            | replay a cada mutação       |
| `preview` | traço ao vivo, fantasma do carimbo, braço do compasso | limpa a cada frame |

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

A grade vem **desativada por padrão** (`board.gridOn = false`) — o quadro
abre limpo, e você liga "mostrar grade" só quando for plotar algo.

### Curvas: cartesianas, paramétricas e polares

Três formas no mesmo painel: `y = f(x)` (1 amostra a cada ~2 px de coluna),
`x(t), y(t)` e `r(t)` (1200 amostras em `t ∈ [t0, t1]`; os campos de faixa
aceitam expressões constantes como `2pi`). Amostras não finitas, pontos
muito além do viewport ou saltos grandes em pixels quebram a polyline em
ramos — `tan(x)` vira uma família de ramos sem espigões verticais, e uma
rosácea `r = 2cos(3t)` fecha perfeitamente.

### Ferramenta de linha (H/V)

Clique em A, arraste até B, solte: traça uma reta perfeitamente horizontal
ou vertical (o eixo de maior deslocamento vence, recalculado a cada
movimento do mouse — dá pra "girar" a prévia entre H e V antes de soltar).
Uma régua, não um traçador de gesto.

Essa ferramenta substituiu o antigo "modo +", que tentava reconhecer o
gesto do mouse em tempo real com uma máquina de estados com histerese
(zona morta de 6 px, corner a cada 12 px de desvio perpendicular). Na
prática exigia controlar a velocidade e a direção do gesto com cuidado pra
não zigzaguear — pouco previsível pra desenhar com mouse. A régua clique→
arraste→solte é direta: você mira o ponto final e pronto.

### Compasso

Clique pra **fixar o centro** (o pino), clique de novo pra **desenhar o
raio** (o lápis) até onde você clicou. O pino continua fixo depois — clique
de novo em qualquer lugar pra traçar outro círculo concêntrico, como um
compasso de verdade riscando vários raios sem tirar a ponta do papel. Pra
recomeçar em outro centro, clique de novo no botão "Compasso" na barra (ou
Esc, que sai pra mãozinha e descarta o pino).

Interação por **cliques independentes**, não por um arraste contínuo: o
primeiro clique só fixa; entre um clique e outro, o braço do compasso
(linha tracejada pino→cursor) e a prévia do círculo seguem o mouse mesmo
sem o botão pressionado (`onHover`, o mesmo mecanismo do fantasma do
carimbo). O círculo em si é uma polyline de 96 segmentos ao redor do
centro (`circlePoints()`, função pura testada isoladamente).

### Mãozinha e Esc

A ferramenta 🖐 só navega (pan), nunca desenha — reaproveita a mesma
`vp.panPx` do pan por botão do meio, então o comportamento é idêntico
independente de como se ativa. **Esc** sempre cancela a ação da ferramenta
atual e troca para a mãozinha, de qualquer modo (caneta, carimbo, compasso...):
um jeito rápido de "sair do modo de escrita" sem precisar acertar um botão
pequeno na barra lateral.

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

O slider de volatilidade vai de 0 a 4 (era 0–1.5 — pouco alcance pra quem
queria linhas bem selvagens) e a seção "Browniano" mora logo abaixo da barra
de ferramentas, não lá embaixo no meio de outros controles, já que é o único
parâmetro específico dessa ferramenta.

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

### Paleta de símbolos

24 botões (gregas, operadores, cálculo/conjuntos) acima do campo de LaTeX;
clicar insere o comando correspondente na posição do cursor dentro do campo
— não só no fim. `selectionStart`/`selectionEnd` de um `<input>` sobrevivem
à perda de foco, então clicar num botão, que tira o foco do campo, ainda
insere exatamente onde o cursor estava. Símbolos que precisam de chaves
(`\sqrt{}`) têm um `caret` próprio na definição, deixando o cursor **dentro**
das chaves depois de inserir, pronto pra digitar o conteúdo.

### Carimbo LaTeX

`POST /api/render_latex {latex, height_px, color}` → PNG transparente em
base64 + dimensões + `rendered: bool`. O mathtext do matplotlib cobre a
maior parte do LaTeX matemático sem nenhum TeX instalado, mas é um
**subconjunto** — não entende `\begin{array}`, `\begin{matrix}` e outros
ambientes. Quando a expressão não parseia, `render_math()` nunca lança
exceção: cai para texto literal (`\mathrm`) e devolve `rendered: false` em
vez de fingir que gerou uma fórmula. O frontend usa esse sinal pra avisar
("isso não é LaTeX que o renderizador entende") e, crucialmente, **não
troca sozinho pra ferramenta Carimbo** quando `rendered: false` — a primeira
versão dessa checagem só mostrava o aviso mas deixava o fluxo seguir igual,
então dava pra carimbar o texto quebrado sem perceber que nada tinha
mudado. Agora, sem `rendered: true`, o app fica parado na ferramenta atual;
carimbar o texto bruto mesmo assim exige escolher "Carimbo" manualmente
(gesto explícito, não acidental). Pede o dobro da altura de exibição e
mostra na metade — nítido em telas hiDPI.

**Carimbo de uso único por padrão.** Depois de posicionar uma fórmula, o
carimbo se descarrega (`stampState.img = null`) — o fantasma some e cliques
seguintes não fazem nada até renderizar de novo. Marcando "permitir
carimbar várias vezes", o carimbo continua carregado e cada clique posiciona
outra cópia (útil pra repetir o mesmo símbolo em vários lugares do quadro).

## OCR de fórmula desenhada (M2) — escondido da UI

O pix2tex (LaTeX-OCR) é bom o bastante em fórmulas *impressas* estruturadas
(frações, somatórios, integrais — testado exaustivamente, ver commits
anteriores), mas fica muito atrás de serviços como o Mathpix em letra
manuscrita de verdade, que é o caso de uso real deste app. Em vez de manter
uma ferramenta que decepciona na maior parte do uso — a razão de ser desta
mudança —, ela foi **removida da barra e do painel lateral**. O código
continua no repositório, intacto:

- `server/ocr.py` — `ocr_image()`, `ink_complexity()` (heurística de
  confiança por contagem de traços) e a integração com `rendered` de
  `mathimg.py` pra detectar quando o modelo alucina algo nem renderizável;
- `server/app.py` — a rota `POST /api/ocr` continua respondendo;
- `static/js/tools/ocr.js`, e `ocrPng()` em `api.js` (frontend) — desconectados
  do `main.js`, mas prontos pra rewire;
- `tests/test_api.py` — os testes de OCR permanecem e passam.

Reativar é reconectar a UI (botão + seção + `handleOcrSelect()`, removidos
de `main.js`) — ou trocar o backend por outro provedor (Mathpix, Google
Vision) reaproveitando o mesmo contrato de `/api/ocr` (`{latex,
low_confidence}`) e o caminho de renderização do carimbo, que não muda.

## Limitações honestas

- O parser conhece uma variável por expressão e funções de um argumento só
  (sem `min(a,b)`, sem `atan2`); nada de variáveis definidas pelo usuário.
- Sem persistência: F5 limpa o quadro (salvar/carregar a cena como JSON é a
  candidata natural pro próximo passo — a cena já é uma lista de objetos
  simples).
- Não dá para selecionar/mover objetos já desenhados; edição é desenhar,
  apagar e desfazer.
- mathtext ≠ TeX completo: ambientes como `\begin{align}` não existem.
- OCR de fórmula manuscrita está fora do escopo atual (ver seção acima) —
  pix2tex não chega perto de serviços comerciais como o Mathpix nesse caso
  de uso específico.

## Roadmap

- **Próximos passos (ideias)** — salvar/carregar a cena (JSON; carimbos
  re-renderizam a partir do LaTeX guardado), selecionar/mover objetos,
  funções de dois argumentos no parser, trocar o OCR por um provedor melhor
  (Mathpix) reaproveitando `/api/ocr`.
