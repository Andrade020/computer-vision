# MathBoard — um paint matemático (m-paint)

Um "Paint" onde as ferramentas são matemáticas: além da caneta comum, você
desenha **só na horizontal/vertical** (modo `+`), transforma retas em
**pontes brownianas**, **plota curvas** `y = f(x)` sobre uma grade estilo
GeoGebra e **carimba fórmulas LaTeX** renderizadas no quadro.

É a fatia **M1** do studio *MathBoard*. A M2 adiciona o caminho inverso:
desenhar a fórmula à mão → OCR (pix2tex) → LaTeX → carimbo bonito.

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
  app.py                         #   POST /api/render_latex + arquivos estáticos
  mathimg.py                     #   LaTeX -> PNG via matplotlib mathtext
                                 #   (vendorizado de handwritten_text/hw/mathimg.py)
static/js/                       # vanilla ES modules — sem build, sem framework
  board.js                       #   3 canvases empilhados + eventos de ponteiro
  scene.js                       #   cena vetorial + undo/redo por comandos
  viewport.js                    #   matemática <-> pixels, grade, ticks 1-2-5
  expr.js                        #   parser de expressões (descida recursiva)
  plot.js                        #   amostragem de y=f(x) -> stroke
  rng.js                         #   mulberry32 + Box-Muller
  api.js                         #   cliente do backend
  tools/{pen,manhattan,brownian,stamp}.js
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

## Limitações honestas (M1)

- **Um único espaço de coordenadas.** Tinta e curvas plotadas viram pixels no
  momento do desenho. Pan (botão do meio) e zoom (roda) movem a grade e
  afetam plots *futuros*; o que já está desenhado fica parado. Cena inteira
  em coordenadas matemáticas é o upgrade planejado da M3.
- O parser conhece uma variável (`x`) e oito funções; sem `sinh`, sem
  paramétricas, sem polares.
- mathtext ≠ TeX completo: ambientes como `\begin{align}` não existem.
- Sem OCR ainda — é exatamente a M2 (pix2tex, CPU, ~65 MB de modelo; treinado
  em fórmula impressa, então manuscrito caprichado funciona melhor).

## Roadmap

- **M2** — desenhar fórmula → recorte → `POST /api/ocr` (pix2tex) → LaTeX →
  carimbo (o caminho de render já existe).
- **M3** — cena em coordenadas matemáticas (pan/zoom movem tudo), mais
  funções, curvas paramétricas e polares.
