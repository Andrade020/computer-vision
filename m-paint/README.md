# MathBoard — um paint matemático (m-paint)

Um "Paint" onde as ferramentas são matemáticas: além da caneta comum, você
desenha **só na horizontal/vertical** (modo `+`), transforma retas em
**pontes brownianas**, **plota curvas** `y = f(x)` sobre uma grade estilo
GeoGebra, **carimba fórmulas LaTeX** renderizadas no quadro — e, com a
ferramenta 🔍, **desenha uma fórmula à mão e ela vira LaTeX** (OCR com
pix2tex), pronta para carimbar bonita no lugar dos rabiscos.

Fatias entregues: **M1** (núcleo) e **M2** (OCR).

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
imagem). Desfazer = remover o último objeto e **redesenhar tudo**. Um snapshot
RGBA de 1600×1000 custaria ~6,4 MB por passo de histórico; redesenhar algumas
centenas de polylines custa menos de 10 ms. Bônus: redimensionar a janela não
perde nada, e `Limpar` também é desfazível (a operação guarda a lista
removida). A borracha é um stroke com `globalCompositeOperation:
'destination-out'` — como o replay preserva a ordem dos acontecimentos, ela
"fura" exatamente o que existia quando você apagou.

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
atom  := NUMBER | 'x' | 'pi' | 'e' | FUNC '(' expr ')' | '(' expr ')'
FUNC  := sin cos tan exp log ln sqrt abs
```

Em vez de montar uma AST e interpretá-la depois, cada função de parse devolve
diretamente um closure `(x) => number` — parser e compilador na mesma passada.
Erros carregam a posição do caractere e aparecem inline na UI.

No plot, uma amostra a cada ~2 px; amostras não finitas ou além de 10× a
meia-altura do viewport **quebram a polyline** — é por isso que `1/x` vira
dois ramos em vez de um espigão vertical na assíntota.

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

## Limitações honestas

- **Um único espaço de coordenadas.** Tinta e curvas plotadas viram pixels no
  momento do desenho. Pan (botão do meio) e zoom (roda) movem a grade e
  afetam plots *futuros*; o que já está desenhado fica parado. Cena inteira
  em coordenadas matemáticas é o upgrade planejado da M3.
- O parser conhece uma variável (`x`) e oito funções; sem `sinh`, sem
  paramétricas, sem polares.
- mathtext ≠ TeX completo: ambientes como `\begin{align}` não existem.
- O OCR é honesto sobre sua origem: treinado em fórmula **impressa**,
  manuscrito caprichado (letras separadas, tamanho generoso) funciona bem
  melhor que garrancho. Nos testes, fórmulas renderizadas e distorcidas
  elasticamente (`x^2+2x+1`, `\frac{a+b}{c}`, `\int_0^1 x\,dx`) voltaram
  perfeitas; a sua letra vai variar.
- A primeira requisição de OCR é lenta (carga do modelo, ~10 s em CPU);
  as seguintes levam ~1–3 s.

## Roadmap

- **M3** — cena em coordenadas matemáticas (pan/zoom movem tudo), mais
  funções, curvas paramétricas e polares.
