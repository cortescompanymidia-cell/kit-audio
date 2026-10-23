# Kit de áudio

Duas ferramentas para trabalhar o som de vídeos como um editor de som: uma que "escuta" e descreve tudo o que acontece no áudio, e outra que trata a voz e faz a mixagem final.

## Instalar

```bash
./instalar.sh
```

Precisa de Python 3.10+ e ffmpeg. Na primeira vez baixa cerca de 3 GB de modelos para a pasta `modelos/`. Nas próximas, só confere.

## 1. Mapa sonoro: descrever o som de um vídeo

```bash
python3 mapa_sonoro.py video.mp4                  # fala em português
python3 mapa_sonoro.py showreel.mp4 --idioma en   # fala em inglês
```

Gera a pasta `video_mapa/` com:

- `mapa.md`: linha do tempo com fala transcrita, efeitos sonoros, eventos da música (subidas e quedas), cortes de imagem e se cada corte cai na batida ou num efeito, além das medidas técnicas.
- `mapa.json`: os mesmos dados, para usar em outros scripts.
- `voz.wav` e `fundo.wav`: a voz e o fundo (música e efeitos) separados, para ouvir cada camada.

O que cada etapa usa:

| Etapa | Modelo |
|---|---|
| Separar voz e fundo | Spleeter (2 faixas) |
| Encontrar trechos de fala | Silero VAD |
| Transcrever | Whisper medium |
| Nomear sons e efeitos | CED base, treinado no AudioSet (527 categorias) |
| Andamento, batidas e energia | librosa |

## 2. Tratar o som: limpar a voz e mixar

```bash
python3 tratar_audio.py gravacao.mp4                        # só a voz
python3 tratar_audio.py gravacao.mp4 --musica trilha.mp3    # voz + música com ducking
```

Cadeia aplicada: corte de graves abaixo de 80 Hz → redução de ruído → remoção de zumbido elétrico (se detectado) → equalização → compressor → de-esser (se necessário) → música com ducking → loudness −14 LUFS com pico real até −1,5 dB.

Opções úteis:

- `--ruido ia | forte | medio | leve`: intensidade da limpeza. O padrão é `ia`. Se a voz ficar artificial, teste `forte` ou `medio`.
- `--alvo -16`: outro loudness final (podcast costuma usar −16).
- `--musica-nivel 8`: quantos dB a música fica abaixo da voz nas pausas.
- `--ducking 10`: quanto a música abaixa a mais enquanto alguém fala.

Gera `gravacao_tratado.wav`, `gravacao_tratado.mp4` (se a entrada for vídeo, sem recodificar a imagem) e `gravacao_relatorio.md` com as medidas de antes e depois.

## Como ler as medidas

| Medida | O que é | Referência |
|---|---|---|
| Loudness (LUFS) | Volume percebido do vídeo inteiro | −14 para redes sociais |
| Pico real (dBTP) | O ponto mais alto, contando o que acontece entre amostras | até −1 |
| Sinal/ruído (dB) | Diferença entre a fala e o silêncio de fundo | acima de 30 é limpo |
| Brilho 5–9 kHz (%) | Quanto da energia está nos agudos onde fica o "s" | voz natural 15 a 30; acima de 35 chia |
| Voz acima do fundo (dB) | Quanto a voz está mais alta que a música durante a fala | 8 a 15 dB |
| Ducking (dB) | Quanto a música abaixa quando alguém fala | valor negativo significa que abaixa |

## Limites

- Os nomes dos efeitos vêm de um classificador automático. Ele acerta bem categorias amplas (impacto, whoosh, aplausos, risadas, ambiente), mas não conhece termos de biblioteca de efeitos como "riser". Num corte, "veículo" costuma ser um whoosh.
- A separação de voz e fundo pode deixar vazar um pouco de uma camada na outra.
- O tratamento é ajustado por medidas. Sempre ouça o resultado: se algo soar estranho, mude o nível de `--ruido`.
