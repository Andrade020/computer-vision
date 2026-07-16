# Classical Filters — filtros classicos de imagem, vetorizados

Transforma o app original (`interactiveinterface.py`, uma GUI Tkinter de
filtros classicos com loops manuais em Python) num **pacote vetorizado +
CLI + GUI polida**: os mesmos efeitos (brilho/contraste, convolucao, ruido
gaussiano, Kuwahara), mas processando a imagem inteira em algumas operacoes
numpy/cv2/scipy em vez de um loop por pixel — sem os limites artificiais
que o loop manual exigia (resize forcado a 400px, aviso de janela > 9 no
Kuwahara), com suporte a cor real na convolucao, tratamento de borda
correto, exportacao de resultado (que simplesmente nao existia antes) e
checagem de erro ao carregar um arquivo invalido.

## Por que esta arquitetura

O app original tinha dois pontos quentes que eram loops `for` puros em
Python por pixel:

- `convolution_filter`: um loop duplo (`for y / for x`) somando
  `kernel_size^2` multiplicacoes por pixel de saida.
- `kuwahara_filter`: um loop duplo com 4 janelas de vizinhanca por pixel,
  cada uma com seu proprio `np.std`.

Ambos sao O(altura x largura x janela²) em Python puro — ordens de
magnitude mais lento que a mesma conta expressa em operacoes vetoriais do
numpy/OpenCV/SciPy, e o motivo pelo qual a GUI original hard-codava um
resize para 400px no carregamento e alertava acima de janela 9 no Kuwahara
("may cause slow processing"). Reescrever os dois como operacoes vetorizadas
remove os dois limites por completo: a mesma imagem em resolucao total, com
janelas grandes, processa em uma fracao do tempo.

- **Convolucao** vira uma unica chamada a `cv2.filter2D`, que ja e
  implementada em C/SIMD e trata bordas corretamente (`BORDER_REFLECT`).
- **Kuwahara** vira somas de regiao via **tabelas de soma de area**
  (*summed-area tables* / imagens integrais): `cumsum` duas vezes da imagem
  (e do seu quadrado) permite ler a soma de qualquer retangulo em O(1), o
  que calcula a media e a variancia de cada um dos 4 quadrantes, para *todos*
  os pixels da imagem de uma vez, com um punhado de operacoes numpy — nao um
  loop por pixel.

## Estrutura

```
imgfilters/
  io.py           load_image(path) / save_image(image, path) -- checa
                  cv2.imread retornando None (arquivo ruim/ausente), que o
                  app original nunca checava
  pointops.py     adjust_brightness_contrast, add_gaussian_noise,
                  resize_image -- ja eram vetorizadas no original, portadas
                  como estavam (so com docstrings limpas)
  convolution.py  convolution_filter(image, kernel, keep_color=False) via
                  cv2.filter2D; KERNELS = {5 presets, os mesmos do combobox
                  original: blur3x3, horizontal/vertical-derivative,
                  sobel-h, sobel-v}
  kuwahara.py     kuwahara_filter(image, window_size) vetorizado via
                  summed-area tables
imgfilter.py      CLI: encadeia qualquer combinacao de operacoes
filters_gui.py    GUI customtkinter: antes/depois, cartoes de opcoes com
                  switch on/off por efeito, recalculo automatico sempre a
                  partir da imagem original, carregar/salvar, Kuwahara em
                  thread separada
assets/
  icon.ico        icone (16/32/48/256px)
  logo.png        mesmo glifo, 512x512, fundo transparente
requirements.txt  numpy, scipy, opencv-python, Pillow, customtkinter
```

## Bugs corrigidos ao portar

1. **Loops manuais O(pixels x janela) em `convolution_filter` e
   `kuwahara_filter`** — motivo dos limites artificiais (resize a 400px,
   aviso de janela > 9). Corrigido vetorizando os dois (`cv2.filter2D` e
   summed-area tables), o que torna os dois limites desnecessarios.
2. **`convolution_filter` sempre convertia para escala de cinza**, mesmo com
   imagem colorida, e sem opcao de manter cor. Agora `keep_color=True`
   aplica o kernel por canal (B/G/R) de forma nativa via `cv2.filter2D`.
3. **Borda preta nao processada** de `kernel_size // 2` pixels ao redor da
   imagem (o loop original comecava/parava cedo demais e o array de saida
   comecava zerado). `cv2.filter2D(..., borderType=cv2.BORDER_REFLECT)`
   trata a borda espelhando a imagem, sem pixels pretos.
4. **Import morto** `from scipy.ndimage import gaussian_filter` no topo do
   arquivo original, nunca usado em lugar nenhum — removido (nenhum modulo
   novo depende de `scipy.ndimage`; veja "Desvios" abaixo sobre por que
   `scipy` ainda aparece no `requirements.txt`).
5. **Sem exportacao de resultado** — a GUI original so exibia o resultado
   num `tk.Label`, nunca escrevia em arquivo. `imgfilters/io.py::save_image`
   + o botao "Salvar como..." da GUI + `-o`/`--output` da CLI resolvem isso.
6. **`cv2.imread` retornando `None` nunca era checado** — um caminho de
   arquivo ruim/ausente quebrava mais adiante, dentro de `resize_image` ou
   do primeiro filtro, com um erro confuso. `load_image` agora levanta
   `FileNotFoundError` (caminho nao existe) ou `ValueError` (arquivo existe
   mas nao decodifica) na hora, com mensagem clara.
7. **Bug de logica encontrado no Kuwahara original, nao listado no
   levantamento inicial**: o loop calculava `idx` (o indice do quadrante de
   menor variancia de brilho), mas a cor de saida era sempre a media de
   `image[tl_y:br_y+1, tl_x:br_x+1]` — a UNIAO dos 4 quadrantes, nao
   `quadrants[idx]`. Ou seja, `idx` era calculado e descartado, e o filtro
   sempre se comportava como um blur de caixa simples, nunca um Kuwahara de
   verdade (que preserva bordas escolhendo o quadrante mais homogeneo). A
   versao vetorizada usa de fato a cor media do quadrante vencedor — ver
   "Checagem de regressao do Kuwahara" abaixo para os numeros que confirmam
   a diferenca.
8. Comentarios/docstrings com vogais cortadas ("adjsts the bright... of the
   imge") — reescritos como comentarios/docstrings normais em ingles/portugues
   claro nos modulos novos.
9. **Efeitos acumulavam em cima do ultimo resultado, nao do original**: a
   GUI original (e a primeira versao deste app) aplicava cada efeito sobre
   `result_image` quando ele ja existia, entao ajustar de novo o brilho
   (por exemplo, beta=10 e depois beta=20 sem resetar) somava os dois em
   vez de substituir -- o resultado dependia da ordem/historico de cliques,
   nao so dos valores atuais dos controles. A GUI atual sempre recalcula a
   partir de `original_image`, aplicando so os efeitos com switch ligado
   no momento, na ordem fixa ajuste -> convolucao -> ruido -> kuwahara --
   mudar um valor produz o mesmo resultado que carregar a imagem e aplicar
   so aquele valor direto.

### Checagem de regressao do Kuwahara

Comparado numa imagem sintetica pequena (30x30, gradiente + tabuleiro de
xadrez) contra duas versoes do loop manual original:

- Contra uma copia **fiel ao bug** do original (usa a uniao dos quadrantes,
  ignorando `idx`): diferenca media absoluta de **27–48** por canal (0–255)
  dependendo do tamanho da janela — confirma que o bug realmente muda o
  resultado de forma significativa.
- Contra uma copia do loop manual **com o bug corrigido** (usa
  `quadrants[idx]`, a semantica pedida nesta tarefa): diferenca media
  absoluta de **0.0–0.57** por canal, com no maximo alguns pixels isolados
  divergindo (empate de variancia entre quadrantes, resolvido por
  `argmin`/ordem de ponto flutuante) — a versao vetorizada reproduz a
  semantica pretendida quase exatamente, nao apenas "proxima o suficiente".

## Uso (linha de comando)

```bash
# instalar dependencias (provavelmente ja presentes no ambiente)
pip install -r requirements.txt

# brilho + contraste
python imgfilter.py entrada.png -o saida.png --brightness 20 --contrast 1.2

# convolucao (5 presets: blur3x3, horizontal-derivative, vertical-derivative,
# sobel-h, sobel-v), em cor (--keep-color) ou em escala de cinza (padrao)
python imgfilter.py entrada.png -o saida.png --conv sobel-h --keep-color

# ruido gaussiano
python imgfilter.py entrada.png -o saida.png --noise 15

# kuwahara (agora vetorizado -- sem limite artificial de janela)
python imgfilter.py entrada.png -o saida.png --kuwahara 9

# encadeando tudo (ordem fixa e sensata: resize -> brilho/contraste ->
# convolucao -> ruido -> kuwahara; so roda o que voce passar como flag)
python imgfilter.py entrada.png -o saida.png --resize 800 \
    --brightness 10 --contrast 1.1 --conv sobel-h --keep-color \
    --noise 5 --kuwahara 7
```

## Interface grafica

```bash
python filters_gui.py
```

Janela customtkinter (cartoes arredondados, sliders com valor ao vivo, tema
claro papel/tinta com a logo do projeto) em vez dos paineis empilhados do
Tkinter puro original. Carregue uma imagem e ligue o switch de qualquer
cartao (Ajuste / Convolucao / Ruido / Kuwahara) que quiser aplicar — nao
existe mais botao "Aplicar": mexer no switch ou em qualquer slider dispara
um recalculo automatico **sempre a partir da imagem original**, na ordem
fixa ajuste -> convolucao -> ruido -> kuwahara, entao mudar por exemplo o
brilho depois de ja ter mexido no ruido aplica os dois direto na imagem
original, sem acumular um efeito em cima do outro. A pre-visualizacao
Antes/Depois atualiza sozinha lado a lado. **Resetar** desliga todos os
switches, devolve os sliders aos valores padrao e restaura a imagem
original — tanto o botao "Resetar" quanto carregar uma nova imagem levam a
esse mesmo estado limpo. **Salvar como...** exporta o resultado atual (o
app original nunca exportava nada). O Kuwahara roda numa thread separada
com uma barra de progresso indeterminada — mesmo vetorizado, ainda e
O(pixels), entao isso evita que a janela pareca travada em imagens grandes
ou janelas grandes; um contador de geracao descarta qualquer resultado que
fique pronto depois que voce ja tenha mudado os controles de novo.

## Limitações honestas

- O Kuwahara vetorizado usa tabelas de soma de area (integral images) em
  vez de `scipy.ndimage.uniform_filter`; isso reproduz exatamente a mesma
  convencao de borda do loop manual original (quadrantes recortados/
  "clampados" na borda, nao refletidos), entao a comparacao com a semantica
  pretendida do original e quase exata (nao apenas "proxima"). Como
  consequencia, `scipy` nao e importado por nenhum modulo em `imgfilters/`
  e foi removido do `requirements.txt`.
- A convolucao em modo colorido (`--keep-color`) aplica o mesmo kernel a
  cada canal B/G/R independentemente — nao ha mistura entre canais (o mesmo
  que qualquer convolucao 2D "ingenua" em RGB; nao tenta, por exemplo,
  trabalhar em luminancia e preservar crominancia separadamente).
- O Kuwahara vetorizado ainda e O(altura x largura) (4 tabelas de soma de
  area + um `argmin` por pixel) — nao ha mais o fator extra de
  `janela²` do loop manual, mas para imagens muito grandes ainda vale a
  pena usar `--resize` antes, ou aguardar a barra de progresso da GUI.
- `resize_image` nunca faz upscale (mantém o comportamento original) —
  passar `--resize` com um valor maior que a imagem atual e um no-op.
- A CLI aplica as operacoes numa ordem fixa (resize, brilho/contraste,
  convolucao, ruido, Kuwahara); para uma ordem diferente, encadeie duas
  chamadas da CLI (arquivo de saida de uma vira entrada da proxima).
- `interactiveinterface.py` e o `README.md` antigo continuam no diretorio,
  intactos — este README e os arquivos novos convivem ao lado deles; a
  remocao dos arquivos antigos fica a criterio de quem revisar isto depois.
