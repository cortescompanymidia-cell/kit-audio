#!/usr/bin/env python3
"""Tratamento de som de voz, como na pós-produção: limpa, equaliza, comprime, mixa com música e finaliza.

Uso:
  python3 tratar_audio.py GRAVACAO.mp4                       # só a voz
  python3 tratar_audio.py GRAVACAO.mp4 --musica trilha.mp3   # voz + música com ducking automático
Opções:
  --ruido  ia | forte | medio | leve   (padrão: ia, que usa o modelo GTCRN; teste "forte" se a voz soar artificial)
  --alvo   loudness final em LUFS (padrão -14, o padrão das redes)
  --musica-nivel  quantos dB a música fica abaixo da voz quando ninguém fala (padrão 8)
  --ducking       quanto a música abaixa durante a fala (padrão 10 dB)
Gera: <nome>_tratado.wav (e <nome>_tratado.mp4 se a entrada for vídeo) e <nome>_relatorio.md.
"""
import argparse, os, sys, subprocess, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import carregar, salvar_wav, tem_video, mono, medir, modelo, vad

SR = 48000


def reduzir_ruido(v, nivel):
    import noisereduce as nr
    if nivel == 'ia':
        import sherpa_onnx as s, librosa
        cfg = s.OfflineSpeechDenoiserConfig(model=s.OfflineSpeechDenoiserModelConfig(
            gtcrn=s.OfflineSpeechDenoiserGtcrnModelConfig(model=modelo('gtcrn_simple.onnx')), num_threads=os.cpu_count() or 4))
        den = s.OfflineSpeechDenoiser(cfg)
        v16 = librosa.resample(v, orig_sr=SR, target_sr=16000)
        out = den.run(v16.tolist(), 16000)
        limpo16 = np.asarray(out.samples, np.float32)[:len(v16)]
        limpo16 = np.pad(limpo16, (0, len(v16) - len(limpo16)))
        # O modelo trabalha em 16 kHz. Para não abafar a voz, calcula-se a máscara (quanto ele manteve de cada
        # frequência a cada instante) e ela é aplicada ao áudio original em 48 kHz. Acima de 8 kHz, onde o modelo
        # não enxerga, usa-se a média da máscara entre 4 e 8 kHz.
        n16, h16 = 512, 128
        V = librosa.stft(v16, n_fft=n16, hop_length=h16); L = librosa.stft(limpo16, n_fft=n16, hop_length=h16)
        mask = np.clip(np.abs(L) / (np.abs(V) + 1e-6), 0, 1)
        n48, h48 = n16 * 3, h16 * 3
        X = librosa.stft(v, n_fft=n48, hop_length=h48)
        T = min(X.shape[1], mask.shape[1]); X, mask = X[:, :T], mask[:, :T]
        f16 = librosa.fft_frequencies(sr=16000, n_fft=n16); f48 = librosa.fft_frequencies(sr=SR, n_fft=n48)
        alto = mask[(f16 >= 4000) & (f16 <= 8000)].mean(0)
        M = np.empty((len(f48), T), np.float32)
        for t in range(T):
            M[:, t] = np.interp(f48, f16, mask[:, t], right=alto[t])
        from scipy.ndimage import uniform_filter
        M = uniform_filter(M, size=(3, 3))
        y = librosa.istft(X * M, hop_length=h48, length=len(v))
        return y.astype(np.float32)
    prop = {'leve': 0.6, 'medio': 0.8, 'forte': 0.95}[nivel]
    return nr.reduce_noise(y=v, sr=SR, stationary=nivel != 'forte', prop_decrease=prop, n_fft=2048).astype(np.float32)


def cadeia_voz(v, zumbido):
    from pedalboard import Pedalboard, HighpassFilter, PeakFilter, HighShelfFilter, Compressor, Gain
    efeitos = [HighpassFilter(cutoff_frequency_hz=80)]
    if zumbido:
        for k in (1, 2, 3, 4):
            efeitos.append(PeakFilter(cutoff_frequency_hz=zumbido * k, gain_db=-30, q=30))
    efeitos += [
        PeakFilter(cutoff_frequency_hz=250, gain_db=-2.5, q=1.0),     # tira o "embolado"
        PeakFilter(cutoff_frequency_hz=4000, gain_db=2.5, q=0.8),     # presença / inteligibilidade
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5),      # ar
        Compressor(threshold_db=-20, ratio=3, attack_ms=5, release_ms=80),
        Gain(gain_db=3),
    ]
    return Pedalboard(efeitos)(v[np.newaxis, :], SR)[0]


def de_esser(v):
    """Suaviza o 's' chiado com o de-esser do ffmpeg."""
    with tempfile.TemporaryDirectory() as d:
        salvar_wav(f'{d}/a.wav', v, SR)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f'{d}/a.wav', '-af', 'deesser=i=0.5:m=0.5:f=0.5', f'{d}/b.wav'], check=True)
        return carregar(f'{d}/b.wav', SR, 1)[0]


def finalizar(x, alvo, pico=-1.5):
    """Loudness e pico real exatos com o loudnorm do ffmpeg em duas passagens (modo linear, sem bombear)."""
    import json, re
    with tempfile.TemporaryDirectory() as d:
        salvar_wav(f'{d}/a.wav', x, SR)
        f1 = f'loudnorm=I={alvo}:TP={pico}:LRA=11:print_format=json'
        e = subprocess.run(['ffmpeg', '-hide_banner', '-i', f'{d}/a.wav', '-af', f1, '-f', 'null', '-'], capture_output=True, text=True).stderr
        m = json.loads(re.search(r'\{[^{}]*\}', e[e.rfind('Parsed_loudnorm'):]).group(0))
        f2 = (f'loudnorm=I={alvo}:TP={pico}:LRA=11:measured_I={m["input_i"]}:measured_TP={m["input_tp"]}:'
              f'measured_LRA={m["input_lra"]}:measured_thresh={m["input_thresh"]}:offset={m["target_offset"]}:linear=true')
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f'{d}/a.wav', '-af', f2, '-ar', str(SR), f'{d}/b.wav'], check=True)
        return carregar(f'{d}/b.wav', SR, x.shape[0] if x.ndim == 2 else 1)


def normalizar(x, alvo):
    import pyloudnorm as pyln
    l = pyln.Meter(SR).integrated_loudness(x.T if x.ndim == 2 else x)
    return x * 10 ** ((alvo - l) / 20)


def curva_ducking(trechos, n, reducao_db, ataque=0.15, soltura=0.5):
    """Ganho da música: abaixa `reducao_db` durante a fala, com rampas suaves de entrada e saída."""
    alvo = np.zeros(n, np.float32)
    for a, b in trechos:
        alvo[int(a * SR):int(b * SR)] = 1
    # rampas: sobe rápido (ataque) e desce devagar (soltura)
    env = np.zeros(n, np.float32); e = 0.0
    ka, ks = 1 / (ataque * SR), 1 / (soltura * SR)
    passo = 256
    for i in range(0, n, passo):
        t = alvo[i]
        e = min(t, e + ka * passo) if t > e else max(t, e - ks * passo)
        env[i:i + passo] = e
    return 10 ** (-reducao_db * env / 20)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('entrada'); ap.add_argument('--musica'); ap.add_argument('--ruido', default='ia', choices=['leve', 'medio', 'forte', 'ia'])
    ap.add_argument('--alvo', type=float, default=-14); ap.add_argument('--musica-nivel', type=float, default=8)
    ap.add_argument('--ducking', type=float, default=10); ap.add_argument('--saida')
    a = ap.parse_args()
    from pedalboard import Pedalboard, Limiter
    import librosa
    base = os.path.splitext(a.entrada)[0] if not a.saida else os.path.splitext(a.saida)[0]
    print('[1/5] medindo o original…', flush=True)
    orig = carregar(a.entrada, SR, 1)[0]
    antes = medir(orig, SR)

    print(f'[2/5] limpando ruído ({a.ruido})…', flush=True)
    v = reduzir_ruido(orig, a.ruido)
    print('[3/5] equalizando e comprimindo a voz…', flush=True)
    usou_deesser = False
    v = cadeia_voz(v, antes['zumbido'])
    if medir(v, SR)['chiado_s_pct'] > 35:
        v = de_esser(v); usou_deesser = True
    v = normalizar(v, a.alvo - 1)
    voz_depois = medir(v, SR)

    mix = v[np.newaxis, :].repeat(2, 0)
    info_musica = ''
    if a.musica:
        print('[4/5] mixando a música com ducking…', flush=True)
        m = carregar(a.musica, SR, 2)
        if m.shape[1] < len(v):  # repete a trilha se for curta
            m = np.tile(m, (1, int(np.ceil(len(v) / m.shape[1]))))
        m = m[:, :len(v)]
        m = normalizar(m, a.alvo - 1 - a.musica_nivel)
        trechos = vad(librosa.resample(v, orig_sr=SR, target_sr=16000))
        g = curva_ducking(trechos, len(v), a.ducking)
        fim = int(SR * 2)  # fade de 2 s no final da música
        g[-fim:] *= np.linspace(1, 0, fim)
        mix = mix + m * g
        info_musica = f'Música {a.musica_nivel:.0f} dB abaixo da voz nas pausas e mais {a.ducking:.0f} dB abaixo durante a fala ({len(trechos)} trechos de fala detectados).'

    print('[5/5] finalizando (loudness e limitador)…', flush=True)
    mix = Pedalboard([Limiter(threshold_db=-3, release_ms=100)])(normalizar(mix, a.alvo).astype(np.float32), SR)
    mix = finalizar(mix, a.alvo)
    depois = medir(mix, SR)
    depois.update({k: voz_depois[k] for k in ('piso_ruido_db', 'relacao_sinal_ruido_db', 'chiado_s_pct', 'zumbido')})
    wav = base + '_tratado.wav'; salvar_wav(wav, mix, SR)
    saidas = [wav]
    if tem_video(a.entrada):
        mp4 = base + '_tratado.mp4'
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', a.entrada, '-i', wav, '-map', '0:v', '-map', '1:a',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-shortest', mp4], check=True)
        saidas.append(mp4)

    linhas = [('Loudness (LUFS)', 'lufs', 'alvo das redes: −14'), ('Pico real (dBTP)', 'pico_dbtp', 'seguro: até −1'),
              ('Piso de ruído (dB)', 'piso_ruido_db', 'quanto mais baixo, mais limpo'),
              ('Sinal/ruído (dB)', 'relacao_sinal_ruido_db', 'acima de 30 é limpo'),
              ('Brilho 5–9 kHz (%)', 'chiado_s_pct', 'voz natural: 15 a 30; acima de 35 chia'), ('Zumbido elétrico', 'zumbido', '')]
    R = [f'# Relatório de tratamento · {os.path.basename(a.entrada)}', '', '| Medida | Antes | Depois | Referência |', '|---|---|---|---|']
    for nome, k, ref in linhas:
        fmt = lambda v_: ('não' if v_ is None else f'{v_} Hz') if k == 'zumbido' else v_
        R.append(f'| {nome} | {fmt(antes[k])} | {fmt(depois[k])} | {ref} |')
    R += ['', '_Piso de ruído, sinal/ruído, chiado e zumbido são medidos só na voz (sem a música)._', '', '**Cadeia aplicada:** corte de graves abaixo de 80 Hz → redução de ruído (' + a.ruido + ')' +
          (f' → remoção de zumbido de {antes["zumbido"]} Hz' if antes['zumbido'] else '') +
          ' → equalização (−2,5 dB em 250 Hz, +2,5 dB em 4 kHz, +1,5 dB acima de 10 kHz) → compressor 3:1' +
          (' → de-esser' if usou_deesser else '') + f' → loudness {a.alvo:.0f} LUFS com pico real até −1,5 dB.', '']
    if info_musica: R += [info_musica, '']
    R += ['Arquivos: ' + ', '.join(os.path.basename(s_) for s_ in saidas)]
    open(base + '_relatorio.md', 'w').write('\n'.join(R) + '\n')
    print('\n'.join(R))


if __name__ == '__main__':
    main()
