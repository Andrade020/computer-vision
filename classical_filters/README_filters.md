# Classical Filters — filtros classicos de imagem, vetorizados, com dominio da frequencia, bordas, morfologia e segmentacao

Transforma o app original (`interactiveinterface.py`, uma GUI Tkinter de
filtros classicos com loops manuais em Python) num **pacote vetorizado +
CLI + GUI polida**: os mesmos efeitos (brilho/contraste, convolucao, ruido
gaussiano, Kuwahara), mas processando a imagem inteira em algumas operacoes
numpy/cv2 em vez de um loop por pixel — sem os limites artificiais
que o loop manual exigia (resize forcado a 400px, aviso de janela > 9 no
Kuwahara), com suporte a cor real na convolucao, tratamento de borda
correto, exportacao de resultado (que simplesmente nao existia antes) e
checagem de erro ao carregar um arquivo invalido. Depois disso, ganhou duas
camadas novas que o original nunca teve: filtragem no **dominio da
frequencia** (FFT 2D) -- low/high/band-pass e um filtro notch para remover
padroes periodicos, com visualizacao do espectro de magnitude da imagem --
e **deteccao de bordas** (gradiente Sobel, Laplaciano/LoG, Canny -- este
ultimo com uma visualizacao opcional de cada estagio interno, nao so o
resultado final), **morfologia matematica** (erosao, dilatacao, abertura,
fechamento, tophat, blackhat) e **limiarizacao/segmentacao** (Otsu,
adaptativo, watershed, k-means por cor).

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

## Por que filtrar tambem no dominio da frequencia

Todo filtro em `convolution.py` trabalha no **dominio espacial** -- olha
para uma vizinhanca pequena de pixels ao redor de cada pixel de saida e
combina com um kernel. Isso e intuitivo, mas algumas ideias sao muito mais
faceis de expressar no **dominio da frequencia**: "mantenha so o conteudo
suave/gradual" (borrar), "mantenha so as bordas nitidas/textura fina"
(realce de detalhe), ou "essa imagem tem um padrao periodico (linhas de
escaneamento, moire, meio-tom) numa frequencia espacial especifica -- tire
so essa" (filtro notch) -- nenhuma das quais tem um kernel espacial obvio,
mas viram uma mascara de uma linha no dominio da frequencia.

A FFT 2D de uma imagem decompõe ela em grades senoidais de toda orientacao
e frequencia possivel; baixa frequencia (perto do centro do espectro
deslocado) corresponde a mudancas de brilho lentas e suaves, e alta
frequencia (perto das bordas do espectro) corresponde a bordas nitidas,
textura fina e ruido. Multiplicar a transformada por uma mascara que zera
uma regiao, e depois transformar de volta, e matematicamente equivalente a
convoluir com algum kernel espacial (possivelmente enorme e impraticavel de
escrever à mão) -- o dominio da frequencia so torna alguns filtros triviais
de expressar e enxergar, em vez de precisar de um kernel desenhado a mão.

Um exemplo concreto e um pouco contraintuitivo, verificado nesta sessao:
um filtro passa-baixa **ideal** (corte duro tipo tudo-ou-nada) parece que
deveria ser a *melhor* versao da ideia, mas sua borda abrupta no dominio da
frequencia produz um "ringing" (ecos fantasmas, ondulacoes visiveis perto
de bordas nitidas) -- o mesmo fenomeno de Gibbs que aparece como artefato
audivel no `compress_audio` ingênuo do projeto irmao `audio_processor`. Um
filtro **gaussiano** (transicao suave, sem borda dura) evita esse ringing
por completo, ao custo de um corte de frequencia menos preciso. Testado
numa imagem sintetica com uma borda nitida (step edge): o filtro ideal
produz um overshoot de **~23 unidades** de brilho (0-255) alem do valor
maximo original da imagem; o filtro gaussiano com o mesmo cutoff produz um
overshoot de **~0** (~1e-9, ruido de ponto flutuante) -- uma demonstracao
numerica limpa do trade-off, nao so uma afirmacao teorica.

## Bordas: tres perguntas diferentes sobre "onde esta a borda"

`gradient_magnitude`, `laplacian_edges` e `canny_edges` respondem a
pergunta "onde tem uma borda?" de tres jeitos diferentes, cada um com uma
definicao diferente do que "borda" significa:

- **Gradiente (Sobel)**: uma borda e onde o brilho muda rapido em alguma
  direcao -- reusa os proprios presets `sobel-h`/`sobel-v` deste projeto
  (`convolution.py`) e combina os dois como `sqrt(gx² + gy²)`. Barato e
  direto, mas produz bandas de borda grossas e borradas, e e sensivel a
  ruido (um unico pixel ruidoso tambem tem "gradiente alto").
- **Laplaciano/LoG**: olha pra derivada *segunda* em vez da primeira --
  uma borda e onde a curvatura do brilho cruza zero. Responde a bordas em
  qualquer direcao com um unico kernel, mas e ainda mais sensivel a ruido
  que o gradiente -- por isso quase sempre vem acompanhado de um borrado
  gaussiano antes (o "G" de "LoG", Laplacian of Gaussian).
- **Canny**: o unico dos tres pensado pra produzir um mapa de bordas
  limpo e fino, nao so uma imagem de "quanto tem de borda aqui". Roda um
  pipeline fixo: borra (reduz ruido) -> calcula o gradiente (magnitude e
  direcao) -> afina pra linhas de 1 pixel mantendo so os maximos locais na
  direcao do gradiente (supressao de nao-maximos) -> decisao de dois
  limiares (histerese: acima do limiar alto sempre fica, entre os dois
  limiares so fica se conectar a uma borda forte). A GUI e a CLI (
  `--canny-stages`) expõem cada estagio intermediario separadamente, ja
  que ver *por que* o Canny escolheu uma borda ensina mais que só ver o
  resultado final.

## Morfologia: erosao/dilatacao e as quatro combinacoes uteis

Onde a convolucao trata cada pixel como uma media ponderada da vizinhanca,
a morfologia trata a vizinhanca ("elemento estruturante") como uma sonda de
forma e faz uma pergunta de minimo/maximo em vez de soma ponderada:
**erosao** troca cada pixel pelo MINIMO da vizinhanca (regioes claras
encolhem), **dilatacao** pelo MAXIMO (regioes claras crescem). Encadear as
duas numa ordem fixa da mais duas operacoes: **abertura** (erode, depois
dilata) remove manchas claras pequenas sem alterar formas maiores;
**fechamento** (dilata, depois erode) preenche buracos escuros pequenos.
**Tophat** (= original - abertura) e **blackhat** (= fechamento - original)
isolam exatamente o que essas duas descartaram -- os detalhes pequenos
claros/escuros, sozinhos contra um fundo quase preto.

Verificado nesta sessao numa imagem sintetica (um quadrado claro de 60x60
com uma mancha clara isolada de 2x2 e um buraco escuro de 3x3 dentro dele):
a abertura remove a mancha isolada por completo (255 -> 0 de brilho) mas
preserva o quadrado maior praticamente intacto (200 -> 200); o fechamento
preenche o buraco escuro (0 -> 200); tophat isola exatamente a mancha
pequena (255 no local da mancha, 0 no centro do quadrado grande) e blackhat
isola exatamente o buraco (200 no buraco, 0 no fundo liso) -- uma
demonstracao numerica limpa de que cada operacao faz exatamente o que a
teoria promete, nao so "parece certo visualmente".

## Limiarizacao e segmentacao: cinco jeitos de agrupar pixels

`otsu_threshold`, `adaptive_threshold`, `connected_components`,
`watershed_segments` e `kmeans_color_segments` respondem "quais pixels
pertencem juntos" de cinco jeitos diferentes:

- **Otsu** escolhe UM limiar global automaticamente -- testa todo corte
  possivel e fica com o que melhor separa o histograma de brilho em dois
  grupos concentrados. Funciona bem com iluminacao uniforme e um histograma
  genuinamente bimodal.
- **Adaptativo** usa um limiar DIFERENTE pra cada regiao (compara cada
  pixel com a media local de uma janela ao redor) -- e o que Otsu nao
  consegue fazer: sob iluminacao desigual (gradiente, sombra), um unico
  corte global classifica regioes inteiras errado, enquanto o adaptativo
  se ajusta a base local. Testado com uma imagem sintetica simulando uma
  pagina escaneada com iluminacao desigual (fundo variando de 60 a 200,
  "tinta" sempre 50 niveis mais escura que o fundo LOCAL, nao um valor
  absoluto fixo): Otsu (limiar global) acertou so 36% da faixa de tinta
  verdadeira (IoU 0.36); o adaptativo acertou 97% (IoU 0.97) -- a mesma
  faixa de tinta e absolutamente ambigua pra qualquer corte global (tinta
  do lado claro tem o mesmo brilho que papel do lado escuro), mas
  perfeitamente resolvivel olhando so a vizinhanca local.
- **Componentes conexos** pega uma mascara ja binaria e agrupa pixels de
  primeiro-plano que se tocam em blobs numerados -- "quantos objetos
  distintos tem, e o tamanho/posicao de cada um".
- **Watershed** resolve o caso mais dificil que componentes conexos nao
  consegue: dois objetos que SE TOCAM (estao literalmente conectados) mas
  deveriam contar como separados. Trata o brilho como um mapa topografico e
  "inunda" a partir de pontos internos confiantes (picos da transformada de
  distancia -- distancia de cada pixel de primeiro-plano ate o fundo mais
  proximo), desenhando uma crista divisora onde duas inundacoes se
  encontrariam. Testado com dois circulos sinteticos que se tocam:
  `connected_components` ve os dois como UM UNICO blob (estao conectados de
  verdade); `watershed_segments` separa corretamente em DOIS objetos,
  desenhando a fronteira exatamente no ponto de contato.
- **K-means por cor** ignora forma/posicao completamente e agrupa pixels só
  por semelhanca de cor -- "reduza essa imagem a k cores representativas."
  Util quando regioes se distinguem melhor por cor do que por brilho ou
  conectividade.

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
  frequency.py    fft2_channel/ifft2_channel (par de FFT 2D + inversa, com
                  DC centralizado via fftshift); low_pass_mask/
                  high_pass_mask/band_pass_mask (kind="ideal"|"gaussian");
                  notch_mask (remove uma frequencia especifica, ex.:
                  padroes periodicos); build_mask (dispatch por nome, como
                  KERNELS); apply_frequency_filter(image, mask,
                  keep_color=False); magnitude_spectrum_image (a "foto"
                  em escala de cinza do espectro, para visualizacao)
  edges.py        gradient_magnitude (Sobel), laplacian_edges (LoG),
                  canny_edges, canny_stages (dict com cada estagio interno:
                  borrado/gradiente/direcao/bordas finais)
  morphology.py   erode/dilate/opening/closing/tophat/blackhat via
                  cv2.erode/cv2.dilate/cv2.morphologyEx; apply_morphology
                  (dispatch por nome, como KERNELS/build_mask)
  segmentation.py otsu_threshold/adaptive_threshold (limiarizacao global/
                  local); connected_components + colorize_labels (agrupa
                  blobs conexos + visualizacao colorida); find_contours +
                  draw_contours; watershed_segments (separa objetos que se
                  tocam); kmeans_color_segments (posteriza por cor)
imgfilter.py      CLI: encadeia qualquer combinacao de operacoes
filters_gui.py    GUI customtkinter: antes/depois, cartoes de opcoes com
                  switch on/off por efeito, recalculo automatico sempre a
                  partir da imagem original, carregar/salvar, Kuwahara em
                  thread separada
assets/
  icon.ico        icone (16/32/48/256px)
  logo.png        mesmo glifo, 512x512, fundo transparente
requirements.txt  numpy, opencv-python, Pillow, customtkinter (sem scipy --
                  ver "Limitações honestas")
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

# filtro passa-baixa (borra, mantem so frequencias baixas) suave (gaussiano,
# sem ringing) com cutoff de 30px, alem de salvar uma visualizacao do
# espectro de magnitude da imagem final
python imgfilter.py entrada.png -o saida.png \
    --freq-filter low-pass --freq-cutoff 30 --freq-kind gaussian \
    --spectrum-out espectro.png

# passa-alta (realca bordas/textura, suprime variacoes suaves de brilho)
python imgfilter.py entrada.png -o saida.png --freq-filter high-pass --freq-cutoff 20

# --explain imprime, antes de processar, o que os parametros escolhidos
# significam na pratica, incluindo o trade-off ringing (ideal) vs suave (gaussiano)
python imgfilter.py entrada.png -o saida.png --freq-filter low-pass --freq-cutoff 15 \
    --freq-kind ideal --explain

# deteccao de bordas: gradiente (Sobel), Laplaciano/LoG, ou Canny
python imgfilter.py entrada.png -o saida.png --edges gradient
python imgfilter.py entrada.png -o saida.png --edges laplacian --edge-blur 1.5
python imgfilter.py entrada.png -o saida.png --edges canny --canny-low 50 --canny-high 150

# Canny com os 4 estagios internos salvos separadamente (nome_blurred.png,
# nome_gradient.png, nome_direction.png, nome_edges.png), nao so o resultado final
python imgfilter.py entrada.png -o saida.png --edges canny --canny-stages estagios/nome

# morfologia: erode/dilate/opening/closing/tophat/blackhat, elemento
# estruturante de 5px (retangulo/elipse/cruz)
python imgfilter.py entrada.png -o saida.png --morph opening --morph-size 5 --morph-shape ellipse

# limiarizacao Otsu (automatica) ou adaptativa (resiste a iluminacao desigual)
python imgfilter.py entrada.png -o saida.png --segment otsu
python imgfilter.py entrada.png -o saida.png --segment adaptive --seg-block-size 25 --seg-c 5

# watershed: separa objetos que se tocam (ex.: duas moedas encostadas) --
# --seg-watershed-ratio maior = seeds mais conservadores (melhor pra objetos
# proximos, pode perder objetos finos)
python imgfilter.py entrada.png -o saida.png --segment watershed --seg-watershed-ratio 0.6

# k-means: posteriza a imagem em k cores representativas
python imgfilter.py entrada.png -o saida.png --segment kmeans --seg-k 5

# depois de segmentar (Otsu/adaptativo), tambem salva os componentes conexos
# coloridos e os contornos detectados
python imgfilter.py entrada.png -o saida.png --segment otsu \
    --components-out componentes.png --contours-out contornos.png

# encadeando tudo (ordem fixa e sensata: resize -> brilho/contraste ->
# convolucao -> filtro de frequencia -> bordas -> morfologia -> ruido ->
# kuwahara -> segmentacao; so roda o que voce passar como flag)
python imgfilter.py entrada.png -o saida.png --resize 800 \
    --brightness 10 --contrast 1.1 --conv sobel-h --keep-color \
    --freq-filter low-pass --freq-cutoff 40 --noise 5 --kuwahara 7
```

## Interface grafica

```bash
python filters_gui.py
```

Janela customtkinter (cartoes arredondados, sliders com valor ao vivo, tema
claro papel/tinta com a logo do projeto) em vez dos paineis empilhados do
Tkinter puro original. Carregue uma imagem e mexa direto em qualquer
slider (Ajuste / Frequencia / Ruido / Kuwahara) ou escolha um preset de
kernel/tipo de filtro /ligue "Manter cor" (Convolucao / Frequencia) — isso
ja liga o switch daquele cartao sozinho, sem precisar liga-lo manualmente
antes. O switch continua existindo para desligar um efeito sem perder o
valor ajustado. Nao existe mais botao "Aplicar": qualquer mudanca dispara
um recalculo automatico **sempre a partir da imagem original**, na ordem
fixa ajuste -> convolucao -> frequencia -> ruido -> kuwahara, entao mudar
por exemplo o brilho depois de ja ter mexido no ruido aplica os dois
direto na imagem original, sem acumular um efeito em cima do outro. A
pre-visualizacao Antes/Depois atualiza sozinha lado a lado.

O cartao "Frequencia (FFT)" tem botoes para o tipo de filtro (Low-pass /
High-pass / Band-pass), botoes para o tipo de corte (Suave = gaussiano,
sem ringing; Duro = ideal, corte preciso mas com ringing -- veja "Por que
filtrar tambem no dominio da frequencia" acima), sliders de cutoff, e um
switch **"Ver espectro (FFT) em vez do resultado"** que troca só a
pre-visualizacao "Depois" pela imagem do espectro de magnitude (o
resultado que seria salvo continua sendo a imagem filtrada de verdade, nao
o espectro -- essa e so uma forma de olhar).

O cartao "Bordas" tem botoes para o metodo (Gradiente / Laplaciano /
Canny), sliders para os limiares do Canny e o pre-borrado, e um switch
**"Ver estagios do Canny"** que troca a pre-visualizacao por uma grade 2x2
com os quatro estagios internos (borrado, gradiente, direcao, bordas
finais) lado a lado -- de novo, so muda o que aparece na tela, o resultado
salvo continua sendo o mapa de bordas final. O cartao "Morfologia" tem
botoes para a operacao (Erodir / Dilatar / Abertura / Fechamento / Tophat /
Blackhat), um seletor de forma do elemento estruturante (elipse/retangulo/
cruz), e sliders de tamanho e numero de iteracoes. O cartao "Segmentacao"
(o ultimo estagio do pipeline, depois do Kuwahara) tem botoes para o metodo
(Otsu / Adaptativo / Watershed / K-means), um switch "Inverter" (objeto
escuro sobre fundo claro), sliders especificos de cada metodo (tamanho do
bloco e constante C do adaptativo, razao da distancia do watershed, numero
de cores do k-means).

**Resetar** desliga todos os
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
- **Filtro de frequencia é grayscale por padrao** (mesma convencao do
  `convolution_filter`): `--freq-keep-color`/"Manter cor" aplica a MESMA
  mascara a cada canal B/G/R independentemente -- nao ha conversao pra um
  espaco de cor tipo YCrCb pra filtrar so luminancia e preservar
  crominancia (o mesmo tipo de simplificacao ja documentado pra
  convolucao colorida).
- **Notch filter (`notch_mask`) nao tem controle na GUI ainda** -- so
  `low-pass`/`high-pass`/`band-pass` estao expostos nos botoes; remover uma
  frequencia especifica (util pra tirar um padrao periodico tipo moire ou
  linhas de scanner) hoje so é acessivel programaticamente
  (`imgfilters.frequency.notch_mask`), nao via CLI nem GUI.
- **`--freq-cutoff2` só importa pro `band-pass`** -- passar/mexer nele com
  low-pass ou high-pass ativo simplesmente nao tem efeito (a CLI/GUI nao
  avisam disso, so ignoram silenciosamente).
- O espectro de magnitude (`magnitude_spectrum_image`, tanto no
  `--spectrum-out` da CLI quanto no switch "Ver espectro" da GUI) é so
  visualizacao -- nao dá pra editar a mascara desenhando nele (diferente
  do editor espectral do projeto irmao `audio_processor`, que permite
  pintar no espectrograma; aqui a mascara so vem dos sliders/botoes de
  low/high/band-pass).
- **Bordas e morfologia sao grayscale por padrao** (mesma convencao de
  `convolution_filter`/filtro de frequencia). `gradient_magnitude` e todas
  as operacoes de morfologia aceitam `keep_color=True` (aplica por canal,
  sem espaco de cor especial); `laplacian_edges` e `canny_edges` NAO
  expõem `keep_color` -- sempre convertem pra escala de cinza primeiro
  (Canny em especial e definido sobre uma imagem de brilho unico, entao
  nao ha uma nocao natural de "Canny por canal separado").
- **`canny_stages` na GUI nao tem rotulo de texto em cada quadrante** da
  grade 2x2 (borrado/gradiente/direcao/bordas) -- a ordem é fixa (sempre
  nessa sequencia, sentido de leitura esquerda-direita/cima-baixo) e
  descrita no README/docstring, mas nao escrita na propria imagem.
- **A visualizacao "direcao" usa uma roda de matiz (hue) HSV**: a cor de
  cada pixel mostra o ANGULO do gradiente ali (nao ha uma unica cor
  "certa" pra cada direcao alem da convencao HSV padrao -- vermelho,
  verde, azul marcam angulos diferentes, nao um significado fisico
  proprio de cada cor).
- **Morfologia com `keep_color=False` (padrao) converte pra escala de
  cinza antes de aplicar** -- diferente de uma erosao/dilatacao "por
  pixel-RGB" que trataria cada canal como uma imagem binaria/grayscale
  separada sem forcar luminancia primeiro; use `keep_color=True` se
  precisar do comportamento por-canal.
- **`--components-out`/`--contours-out` esperam uma mascara ja binaria**
  (de `--segment otsu`/`adaptive`) -- se usados sem `--segment` (ou com
  `--segment watershed`/`kmeans`, que nao produzem mascara binaria), a CLI
  cai para escala de cinza automaticamente pra nao travar, mas o resultado
  ("componentes conexos" de uma foto normal) tende a nao fazer sentido —
  quase todo pixel nao-preto conta como "primeiro plano".
- **Watershed precisa de objetos claros sobre fundo escuro (ou vice-versa
  com `--seg-invert`)** com contraste razoavel -- baseia-se inteiramente
  no Otsu interno pra achar o primeiro plano bruto antes da transformada de
  distancia; uma imagem sem separacao clara de brilho entre objeto e fundo
  nao vai segmentar bem, nao importa o `--seg-watershed-ratio` escolhido.
- **K-means por cor nao usa posicao/vizinhanca**, só a cor RGB de cada
  pixel isoladamente -- duas regioes desconexas da mesma cor (ex.: dois
  objetos vermelhos em lados opostos da imagem) caem no MESMO cluster, ao
  contrario de `connected_components`/`watershed`, que exigem contiguidade
  espacial.
- **`cv2.setRNGSeed` em `kmeans_color_segments`** torna os clusters
  reprodutiveis entre chamadas com a mesma imagem/k, mas é um seed GLOBAL
  do RNG interno do OpenCV -- chamar outras funcoes de cv2 que tambem usam
  aleatoriedade entre duas chamadas de `kmeans_color_segments` pode alterar
  o resultado (nao é um problema no uso normal deste app, que so chama
  k-means isoladamente).
