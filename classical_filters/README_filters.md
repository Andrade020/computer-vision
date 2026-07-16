# Classical Filters — filtros classicos de imagem, vetorizados, com filtro no dominio da frequencia

Transforma o app original (`interactiveinterface.py`, uma GUI Tkinter de
filtros classicos com loops manuais em Python) num **pacote vetorizado +
CLI + GUI polida**: os mesmos efeitos (brilho/contraste, convolucao, ruido
gaussiano, Kuwahara), mas processando a imagem inteira em algumas operacoes
numpy/cv2 em vez de um loop por pixel — sem os limites artificiais
que o loop manual exigia (resize forcado a 400px, aviso de janela > 9 no
Kuwahara), com suporte a cor real na convolucao, tratamento de borda
correto, exportacao de resultado (que simplesmente nao existia antes) e
checagem de erro ao carregar um arquivo invalido. Depois disso, ganhou uma
camada nova que o original nunca teve: filtragem no **dominio da
frequencia** (FFT 2D) -- low/high/band-pass e um filtro notch para remover
padroes periodicos, com visualizacao do espectro de magnitude da imagem.

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

# encadeando tudo (ordem fixa e sensata: resize -> brilho/contraste ->
# convolucao -> filtro de frequencia -> ruido -> kuwahara; so roda o que
# voce passar como flag)
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
