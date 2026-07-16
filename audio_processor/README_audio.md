# Audio DSP Studio — trim, compressao de Fourier, eco, reverb, EQ real, espectro, espectrograma e edicao espectral

Reconstrucao do protótipo original (`audiointerface.py`, uma unica tela
Tkinter) num **mini-app organizado**: a matematica de DSP (que ja estava
correta) foi extraida para um pacote testavel e sem GUI (`audiodsp/`), e por
cima dele existem dois front-ends que reusam exatamente o mesmo codigo — uma
CLI (`audioprocess.py`) e uma interface grafica em customtkinter
(`audio_gui.py`).

## Por que esta arquitetura

O prototipo original misturava tudo num arquivo so: as funcoes de DSP, o
matplotlib e a classe Tkinter estavam todas acopladas, o que tornava
impossivel testar a matematica sem abrir uma janela, e reproduzia audio via
`os.startfile(...)` (abre o tocador padrao do SO), que so funciona no
Windows e nao permite parar a reproducao, comparar original vs. processado,
ou tocar o audio sem gravar um arquivo temporario primeiro.

A escolha aqui foi separar em camadas:

1. **`audiodsp/`** — pacote puro numpy/scipy/soundfile/sounddevice, sem
   `tkinter` nem `matplotlib` importados dentro dele (a excecao é
   `sounddevice`, que é necessario para tocar audio de verdade). Cada efeito
   é uma funcao pura `(audio, sr, ...) -> audio`, entao pode ser testada
   isoladamente com um seno sintetico, sem abrir nenhuma janela.
2. **`audioprocess.py`** (CLI) — usa o pacote acima e so importa
   `matplotlib` para o `--spectrum`, exclusivamente na camada de CLI.
3. **`audio_gui.py`** — interface customtkinter que tambem so usa o pacote
   acima; embute o mesmo plot de espectro/espectrograma com
   `FigureCanvasTkAgg`, mas roda os efeitos numa thread separada (reverb em
   audios longos pode demorar) e toca audio direto de um array numpy via
   `sounddevice`, sem arquivo temporario nem depender do tocador padrao do
   sistema. Cada efeito tem um switch on/off; mexer em qualquer switch ou
   slider recalcula automaticamente **a partir do audio original** (nao
   acumula um efeito em cima do outro) e redesenha a visualizacao ativa
   sozinho, sem botao "Aplicar". Um `Player` (`audiodsp/playback.py`) da
   transporte de verdade — Play/Pause, Stop, seek e leitura de posicao — em
   vez de um "tocar" disparar-e-esquecer.

## Do espectro ao espectrograma: por que existem os dois

`spectrum.py` responde "quais frequencias aparecem nesse audio, no total?" —
uma unica FFT do sinal inteiro. Isso descarta **quando** cada frequencia
aconteceu: uma nota tocada no inicio e a mesma nota tocada no final parecem
identicas numa FFT do sinal inteiro.

`stft.py` responde "quais frequencias aparecem, e quando?" deslizando uma
janela curta de analise ao longo do audio e tirando a FFT de cada pedaco.
Empilhando essas FFTs lado a lado (frequencia num eixo, tempo no outro) voce
tem um **espectrograma** — literalmente uma foto do som ao longo do tempo. A
GUI expoe os dois lados a lado no cartao "Analise": botoes **Espectro** /
**Espectrograma** trocam a visualizacao (sem precisar recalcular o audio).

Esse fatiamento tambem carrega o trade-off classico da area (uma prima da
mesma ideia de incerteza tempo-frequencia da fisica): uma **janela longa**
(`n_fft` grande) enxerga varios ciclos de uma frequencia baixa, entao
distingue duas notas proximas com precisao — boa resolucao de frequencia. Mas
borra tudo que acontece *dentro* dessa janela, entao dois cliques a 5ms de
distancia viram um borrao so — resolucao de tempo ruim. Uma **janela curta**
faz o oposto: acerta bem o "quando", mas confunde frequencias proximas. Nao
ha almoco gratis, so um dial — os sliders "tamanho da janela" e "sobreposicao"
no cartao "Espectrograma" da GUI (e `--n-fft`/`--hop` na CLI) sao esse dial.

## Editor espectral: pintar diretamente no tempo-frequencia

Se um espectrograma é uma foto do som, o editor espectral (cartao "Editor
Espectral" na GUI) deixa voce **desenhar em cima dessa foto** e ouvir o
resultado. Um efeito comum (trim/eco/reverb/etc.) so enxerga o eixo do
tempo — ele nao consegue expressar "apague só essa faixa de frequencia,
só entre 1.2s e 1.5s". Um efeito no dominio da frequencia (`--compress`)
so enxerga o eixo da frequencia — ele nao consegue expressar "só nesse
trecho de tempo". O espectrograma tem os dois eixos ao mesmo tempo, entao
uma regiao pintada nele pode ser tao especifica quanto "essa frequencia,
nesse instante" — apagar uma tosse, isolar um assobio, silenciar um zumbido
de 60Hz sem tirar o resto do audio.

Mecanicamente: **Iniciar edicao** tira uma STFT do audio atual e guarda uma
"mascara" do mesmo tamanho (tudo 1.0 = nao mexeu em nada). Arrastar o mouse
no espectrograma pinta um retangulo de tempo/frequencia (do tamanho do
"pincel") nessa mascara com o ganho do modo ativo — **Apagar** escreve 0.0
(silencia aquele pedaco), **Realcar** escreve 2.5 (amplifica). **Aplicar**
multiplica a STFT original pela mascara pintada e faz o caminho de volta
(ISTFT) para gerar o audio novo; **Cancelar** descarta a mascara sem tocar
no audio; **Limpar** zera a mascara de volta pra 1.0 sem sair do modo de
edicao. Por ser um desenho livre (nao um slider parametrico), a edicao
espectral é uma operacao manual de uma vez só (como uma ferramenta de
pintura), nao parte do pipeline automatico de Trim/Compressao/Eco/Reverb —
ver "Limitações honestas" abaixo para a consequencia disso.

## EQ real (biquad/IIR): o outro jeito de mexer em frequencia

O editor espectral acima resolve "essa frequencia, só nesse instante" --
mas exige a gravação inteira analisada de antemão (uma STFT). Um
equalizador de verdade resolve um problema diferente: "sempre atenue/realce
essa faixa, para sempre, amostra a amostra, sem precisar olhar o áudio
inteiro antes" -- é assim que todo EQ analógico, todo filtro de sintetizador
e todo efeito de áudio em tempo real funciona, porque um **biquad** (o bloco
básico de um filtro IIR) só olha as 2 últimas amostras de entrada e as 2
últimas de saída para calcular a próxima -- nenhum buffer, nenhuma
antecipação. `audiodsp/filters.py` implementa as fórmulas clássicas do
"Audio EQ Cookbook" (Robert Bristow-Johnson) para os 7 tipos padrão
(lowpass/highpass/bandpass/notch/peak/lowshelf/highshelf); o cartão
"Equalizador" da GUI expõe um EQ de 3 bandas fixo (Graves=shelf, Médios=
peak, Agudos=shelf) construído em cima disso, e o botão **"Resposta EQ"**
no painel "Analise" plota a resposta em frequência combinada das bandas
ativas -- calculada analiticamente a partir dos coeficientes do filtro, sem
precisar do áudio, então atualiza instantaneamente ao mexer num slider.

Verificado nesta sessão com números limpos: uma banda `peak` pedida com
+12dB de ganho mediu +12.16dB no áudio processado de verdade; -12dB mediu
-11.83dB; os shelves de +10dB bateram em +9.99dB (grave) e +9.92dB (agudo)
na regiao afetada, e ~0dB na regiao NAO afetada -- a implementação bate com
a teoria, não só "parece certo".

## Estrutura

```
audiodsp/
  __init__.py    reexporta os submodulos do pacote
  io.py          load_audio(path) -> (audio, sr) com mixagem para mono e
                 tratamento de erro (arquivo corrompido/inexistente nao
                 derruba o programa); save_audio(audio, sr, path)
  effects.py     trim_audio, compress_audio, add_echo, add_reverb -- mesma
                 matematica do prototipo original, docstrings limpas
  spectrum.py    magnitude_spectrum(audio, sr) -> (freqs, mags), so a metade
                 positiva do espectro (0 a sr/2); sem matplotlib, testavel
                 sem display
  stft.py        stft/istft (par de transformada de tempo curto + inversa,
                 com reconstrucao overlap-add por tabela de janela ao
                 quadrado), spectrogram_db (STFT -> magnitude -> dB),
                 paint_region (pinta um retangulo tempo/frequencia de uma
                 mascara com um ganho constante -- o primitivo por tras da
                 edicao espectral) e describe_params (texto didatico sobre
                 os parametros escolhidos); tudo headless, docstrings
                 explicam o trade-off tempo x frequencia
  filters.py     biquad_lowpass/highpass/bandpass/notch/peak/lowshelf/
                 highshelf (formulas do Audio EQ Cookbook); build_biquad
                 (dispatch por nome); apply_filter (via scipy.signal.lfilter
                 -- IIR causal, com resposta de fase real, como um EQ de
                 verdade); apply_eq_bands (cascateia varias bandas, tipo EQ
                 grafico); frequency_response (resposta analitica em dB, via
                 scipy.signal.freqz); describe_filter (texto didatico)
  playback.py    Player (play/pause/resume/seek/posicao) + play(audio, sr) /
                 stop() via sounddevice -- substitui as duas chamadas a
                 os.startfile do prototipo original
audioprocess.py  CLI (argparse): aplica os efeitos pedidos em ordem fixa
                 (trim -> compress -> eco -> reverb -> EQ -> regioes
                 espectrais) e opcionalmente plota o espectro de magnitude,
                 o espectrograma (STFT) e/ou a resposta em frequencia do EQ
                 do resultado final em PNG; --eq/--spectral-region sao os
                 equivalentes roteirizaveis do EQ/pincel da GUI; --explain
                 imprime a explicacao didatica dos parametros de STFT
                 escolhidos
audio_gui.py     interface grafica customtkinter: cartoes "Arquivo",
                 "Efeitos" (switch on/off por efeito, recalculo automatico
                 sempre a partir do original), "Equalizador" (3 bandas fixas
                 -- graves/medios/agudos -- switch+sliders de freq/ganho por
                 banda), "Espectrograma" (sliders de tamanho de janela/
                 sobreposicao + escolha de janela, so afetam a visualizacao,
                 nao o audio), "Editor Espectral" (pincel de tempo/
                 frequencia, modos Apagar/Realcar, Iniciar/Limpar/Cancelar/
                 Aplicar), "Reproducao" (transporte Play/Pause/Stop/seek,
                 alternando entre Original e Processado -- o A/B que faltava
                 no prototipo) e um painel "Analise" com botoes Espectro/
                 Espectrograma/Resposta EQ, atualizado sozinho a cada
                 recalculo
assets/
  icon.ico       icone multi-resolucao (16/32/48/256px), glifo de forma de
                 onda estilizado na paleta tinta/papel
  logo.png       mesmo glifo, 512x512, fundo transparente, usado no
                 cabecalho da janela
requirements.txt numpy, scipy, matplotlib, soundfile, sounddevice,
                 customtkinter, Pillow (tkinter NAO entra aqui -- vem com o
                 Python padrao, nao e um pacote pip)
```

## Uso

### Interface grafica

```bash
python audio_gui.py
```

Clique **Carregar audio...** e mexa direto em qualquer slider de um efeito
(Trim / Compressao / Eco / Reverb / Equalizador) — isso ja liga o switch
daquele efeito sozinho, sem precisar ligar o switch primeiro para so depois
poder mexer no valor. O switch continua existindo para desligar um efeito sem perder o
valor ajustado. Qualquer mudanca dispara, apos um pequeno debounce
(~300ms), um recalculo automatico **sempre a partir do audio original** —
nunca em cima do resultado anterior, entao ajustar por exemplo o ganho do
eco depois de ja ter mexido no trim aplica os dois direto no original, sem
acumular. O cartao "Equalizador" tem 3 bandas fixas (Graves/Medios/Agudos,
cada uma com switch + slider de frequencia + slider de ganho em dB) que
entram no mesmo pipeline automatico, como um EQ de verdade (biquad/IIR),
nao um efeito no dominio da frequencia via STFT. A visualizacao ativa
(Espectro, Espectrograma, ou Resposta EQ, escolhida pelos botoes no topo do
painel "Analise") é redesenhada sozinha a cada recalculo,
sem precisar clicar em nada. O cartao "Espectrograma" tem sliders de tamanho
de janela e sobreposicao, alem da escolha da funcao de janela — mexer neles
so redesenha a visualizacao (nao recalcula o audio) e só faz efeito quando
"Espectrograma" é a visao ativa.

O cartao "Editor Espectral" liga o modo de pintura: escolha **Apagar** ou
**Realcar**, ajuste o tamanho do pincel (em Hz e em ms), clique **Iniciar
edicao** (troca para a visao Espectrograma automaticamente e trava os
controles de STFT, já que mudar o tamanho da janela no meio de uma edicao
invalidaria a mascara em andamento) e arraste no espectrograma. **Limpar**
zera a mascara sem sair do modo de edicao; **Cancelar** sai sem tocar no
audio; **Aplicar** gera o audio novo a partir da mascara pintada. Trocar de
visualizacao ou mexer num efeito parametrico enquanto uma edicao esta em
andamento cancela a edicao automaticamente (o audio por baixo mudou, entao
a mascara em andamento nao faz mais sentido).

O cartao "Reproducao" tem transporte de verdade: Play/Pause
(o mesmo botao alterna), Stop, uma barra de progresso arrastavel (seek) e
os botoes **Processado** / **Original** decidem qual dos dois buffers toca
— se o audio estiver tocando quando voce muda um efeito ou troca de
Processado para Original (ou vice-versa), a reproducao continua do mesmo
ponto no novo buffer, em vez de parar. **Resetar** desliga todos os
efeitos, devolve os sliders aos valores padrao e restaura o audio
exatamente como foi carregado. **Salvar processado como...** abre o
dialogo de salvar arquivo.

### Linha de comando

```bash
# so cortar os primeiros 10s
python audioprocess.py entrada.wav -o saida.wav --trim 10

# compressao de Fourier (mantem 50% dos bins de cada extremidade do espectro)
python audioprocess.py entrada.wav -o saida.wav --compress 0.5

# eco com atraso de 0.5s e ganho 0.6
python audioprocess.py entrada.wav -o saida.wav --echo-delay 0.5 --echo-gain 0.6

# reverb: 10 copias defasadas de 0.05s cada
python audioprocess.py entrada.wav -o saida.wav --reverb-delays 10 --reverb-time 0.05

# tudo junto, na ordem fixa trim -> compress -> eco -> reverb, alem de
# salvar um PNG do espectro de magnitude do resultado final
python audioprocess.py entrada.wav -o saida.wav --trim 10 --compress 0.5 \
    --echo-delay 0.5 --echo-gain 0.6 --reverb-delays 10 --reverb-time 0.05 \
    --spectrum espectro.png

# espectrograma (STFT) do resultado final, com janela/sobreposicao escolhidas
python audioprocess.py entrada.wav -o saida.wav \
    --spectrogram espectrograma.png --n-fft 1024 --hop 256 --window hann

# --explain imprime, antes de processar, o que esses numeros significam na
# pratica (ms de janela, % de sobreposicao, resolucao em Hz e em ms)
python audioprocess.py entrada.wav -o saida.wav --spectrogram espectrograma.png --explain

# edicao espectral roteirizada: apaga 1100-1300Hz entre 1.0 e 2.0 segundos
# (equivalente ao pincel "Apagar" da GUI, sem precisar de mouse) -- repita
# a flag para pintar mais de uma regiao na mesma chamada
python audioprocess.py entrada.wav -o saida.wav \
    --spectral-region 1.0,2.0,1100,1300,0.0

# realca (2x) a faixa 2000-4000Hz do audio inteiro
python audioprocess.py entrada.wav -o saida.wav \
    --spectral-region 0,999,2000,4000,2.0

# EQ real (biquad/IIR): corta 1000Hz em -12dB e realca os graves (abaixo de
# 300Hz) em +6dB -- repita --eq para cascatear mais bandas (tipo EQ grafico)
python audioprocess.py entrada.wav -o saida.wav \
    --eq peak,1000,1.0,-12 --eq lowshelf,300,0.707,6

# remove um zumbido de 60Hz sem tocar no resto (equivalente em tempo real
# ao editor espectral -- mas aplicado sempre, nao so num trecho de tempo)
python audioprocess.py entrada.wav -o saida.wav --eq notch,60,10,0

# --response-out plota a resposta em frequencia combinada das bandas de EQ
# (analitica, nao precisa do audio) -- util pra conferir o EQ antes de ouvir
python audioprocess.py entrada.wav -o saida.wav \
    --eq peak,1000,1.0,-12 --eq lowshelf,300,0.707,6 --response-out resposta.png
```

Cada flag de efeito so e aplicada se voce a passar (nenhum efeito roda por
padrao); `--echo-delay`/`--echo-gain` ativam o eco juntos ou separados
(usando o valor padrao do outro), o mesmo vale para
`--reverb-delays`/`--reverb-time`. `--spectrum` e `--spectrogram` sao
independentes -- pode pedir os dois na mesma chamada. `--eq` roda depois de
trim/compress/echo/reverb e antes de `--spectral-region` (que continua
usando os mesmos `--n-fft`/`--hop`/`--window` do `--spectrogram`).

## Limitações honestas

- **Formatos de audio**: depende do que `soundfile`/libsndfile suportam
  nativamente (WAV, FLAC, OGG, AIFF...). MP3 funciona em muitas instalacoes
  mas nao e garantido em todas as plataformas -- WAV e o formato mais
  confiavel.
- **`compress_audio`** é uma compressao de Fourier ingênua (zera bins do
  meio do espectro), nao um filtro com janela/rampa -- em `p` pequeno pode
  introduzir artefatos audiveis tipo "ringing" (efeito Gibbs), como no
  prototipo original. Isso foi mantido de proposito (é a mesma matematica),
  nao "corrigido" para um filtro mais suave.
- **`add_reverb`** com `num_delays` grande e clipes longos é O(num_delays *
  len(audio)) em Python/numpy puro -- perceptivel na GUI (por isso roda em
  thread separada), mas nao é um algoritmo de reverb de convolucao real.
- **Sem normalizacao/clipping**: eco e reverb somam copias do sinal sem
  normalizar o pico resultante; um audio ja proximo do full-scale pode
  saturar (`sf.write` nao clipa silenciosamente, mas o valor pode estourar o
  intervalo esperado do formato). Nao ha um limitador automatico.
- **Reproducao via `sounddevice`**: precisa de um dispositivo de saida de
  audio disponivel no SO. Em ambientes sem placa de som/servidor de audio
  (ex.: alguns containers/CI) `playback.play()` pode falhar ao abrir o
  stream -- isso nao afeta o processamento em si (CLI e os modulos de
  `audiodsp/` nao dependem de dispositivo de audio nenhum).
- **GUI nao acumula historico de efeitos**: cada recalculo parte sempre do
  audio original e reaplica os efeitos atualmente ligados na ordem fixa
  trim -> compress -> eco -> reverb -- nao ha um historico tipo "desfazer"
  passo a passo, so o estado atual dos switches/sliders (use "Resetar" para
  voltar tudo ao audio como foi carregado).
- **Recalculo automatico com debounce**: arrastar um slider dispara o
  recalculo ~300ms depois que voce solta/para, para nao rodar o pipeline
  inteiro a cada pixel de movimento do mouse; um contador de geracao
  descarta qualquer resultado que fique pronto depois que voce ja tenha
  mudado os controles de novo (evita a UI "voltar" para um estado antigo).
- Os icones/logo (`assets/`) sao um glifo geometrico simples gerado por
  script (PIL), nao uma arte desenhada a mao -- so para o app parecer
  finalizado, sem pretensao de identidade visual elaborada.
- **Eixo de frequencia linear**: tanto o espectrograma quanto o editor
  espectral usam um eixo de frequencia linear, nao log/mel -- para audio
  musical um eixo log costuma ser mais legivel (mais espaço visual pras
  frequencias graves, onde a percepcao humana é mais sensivel a pequenas
  diferenças). Ainda nao foi adicionado. O piso de -80dB e o eixo de cor sao
  fixos (nao ha slider de faixa dinamica ainda).
- **Edicao espectral é uma operacao manual "de uma vez", nao parte do
  pipeline automatico**: diferente de Trim/Compressao/Eco/Reverb (que
  recalculam sozinhos sempre que voce mexe num slider), uma edicao espectral
  so acontece quando voce clica **Aplicar**, e vira o novo `self.audio` --
  ela nao fica "lembrada" como um efeito ligado. Consequencia pratica: se
  voce aplicar uma edicao espectral e DEPOIS mexer num efeito parametrico
  (ex.: o slider do eco), o recalculo automatico desses efeitos parte sempre
  do audio ORIGINAL (sem a edicao espectral) e a pintura manual se perde --
  o app cancela a sessao de edicao automaticamente nesse caso, mas nao
  reaplica a mascara depois. Ordem recomendada: ajuste os efeitos
  parametricos primeiro, edite o espectro por ultimo.
- **Pincel retangular, nao free-form real**: cada arrasto do mouse pinta uma
  serie de retangulos de tempo/frequencia (do tamanho do "pincel") ao longo
  do caminho -- funciona bem na pratica mas nao é uma mascara de forma livre
  pixel-a-pixel; brushes muito grandes com movimento rapido do mouse podem
  deixar saltos entre um retangulo pintado e o proximo.
- **A janela/sobreposicao ficam travadas durante uma edicao**: a mascara
  pintada tem o formato exato da STFT usada para inicia-la, entao mudar
  `n_fft`/sobreposicao/janela no meio invalidaria a pintura -- os controles
  correspondentes ficam desabilitados (cinza) enquanto uma sessao de edicao
  esta ativa, e voltam ao normal quando voce aplica ou cancela.
- **`istft` reconstrói quase exatamente** (`stft` seguido de `istft` bate com
  o original a ~1e-15 de erro relativo em teste com seno sintetico), mas isso
  vale para a janela `hann` com sobreposicao >= 50% usada por padrao; janelas/
  sobreposicoes muito incomuns podem nao satisfazer a condicao COLA
  (constant-overlap-add) e reconstruir com mais erro perto das bordas.
- **EQ da GUI é fixo em 3 bandas** (Graves=shelf, Médios=peak, Agudos=shelf,
  com Q fixo em cada uma) -- não dá pra escolher o tipo de cada banda nem
  adicionar mais bandas pela interface gráfica; a CLI (`--eq`, repetível,
  qualquer um dos 7 tipos) é mais flexível nesse sentido.
- **Filtro IIR tem fase de verdade** (diferente do `istft`/`filtfilt`, que
  são zero-fase): perto da frequência de corte, o biquad desloca levemente
  o sinal no tempo, exatamente como um EQ analógico faria -- não é um bug,
  é uma propriedade genuína de filtros causais (só olham amostras passadas).
- **Sem normalização de ganho ao cascatear bandas**: `apply_eq_bands` só
  encadeia os filtros; se várias bandas realçarem frequências que se somam,
  o pico do sinal pode passar de 0dBFS (mesma limitação já documentada para
  eco/reverb -- não há limitador automático).
