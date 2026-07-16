# Audio DSP Studio — trim, compressao de Fourier, eco, reverb e espectro

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
   acima; embute o mesmo plot de espectro com `FigureCanvasTkAgg`, mas roda
   os efeitos numa thread separada (reverb em audios longos pode demorar) e
   toca audio direto de um array numpy via `sounddevice`, sem arquivo
   temporario nem depender do tocador padrao do sistema. Cada efeito tem um
   switch on/off; mexer em qualquer switch ou slider recalcula
   automaticamente **a partir do audio original** (nao acumula um efeito em
   cima do outro) e redesenha o espectro sozinho, sem botao "Aplicar". Um
   `Player` (`audiodsp/playback.py`) da transporte de verdade — Play/Pause,
   Stop, seek e leitura de posicao — em vez de um "tocar" disparar-e-esquecer.

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
  playback.py    Player (play/pause/resume/seek/posicao) + play(audio, sr) /
                 stop() via sounddevice -- substitui as duas chamadas a
                 os.startfile do prototipo original
audioprocess.py  CLI (argparse): aplica os efeitos pedidos em ordem fixa
                 (trim -> compress -> eco -> reverb) e opcionalmente plota
                 o espectro final em PNG
audio_gui.py     interface grafica customtkinter: cartoes "Arquivo",
                 "Efeitos" (switch on/off por efeito, recalculo automatico
                 sempre a partir do original), "Reproducao" (transporte
                 Play/Pause/Stop/seek, alternando entre Original e
                 Processado -- o A/B que faltava no prototipo) e um painel
                 de espectro embutido, atualizado sozinho a cada recalculo
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
(Trim / Compressao / Eco / Reverb) — isso ja liga o switch daquele efeito
sozinho, sem precisar ligar o switch primeiro para so depois poder mexer no
valor. O switch continua existindo para desligar um efeito sem perder o
valor ajustado. Qualquer mudanca dispara, apos um pequeno debounce
(~300ms), um recalculo automatico **sempre a partir do audio original** —
nunca em cima do resultado anterior, entao ajustar por exemplo o ganho do
eco depois de ja ter mexido no trim aplica os dois direto no original, sem
acumular. O espectro é redesenhado sozinho a cada recalculo, sem precisar
clicar em nada. O cartao "Reproducao" tem transporte de verdade: Play/Pause
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
```

Cada flag de efeito so e aplicada se voce a passar (nenhum efeito roda por
padrao); `--echo-delay`/`--echo-gain` ativam o eco juntos ou separados
(usando o valor padrao do outro), o mesmo vale para
`--reverb-delays`/`--reverb-time`.

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
