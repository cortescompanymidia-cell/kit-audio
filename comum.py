"""Funções compartilhadas pelo kit de áudio: carregar áudio, medir volume e acessar os modelos."""
import os, glob, subprocess, numpy as np

KIT = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(KIT, 'modelos')


def carregar(caminho, sr=44100, canais=2, inicio=None, duracao=None):
    """Lê qualquer áudio ou vídeo via ffmpeg. Retorna array (canais, amostras) float32."""
    cmd = ['ffmpeg', '-loglevel', 'error']
    if inicio is not None: cmd += ['-ss', str(inicio)]
    if duracao is not None: cmd += ['-t', str(duracao)]
    cmd += ['-i', caminho, '-vn', '-ac', str(canais), '-ar', str(sr), '-f', 'f32le', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, canais).T.copy()


def salvar_wav(caminho, x, sr):
    import soundfile as sf
    x = np.asarray(x, np.float32)
    sf.write(caminho, x.T if x.ndim == 2 else x, sr, subtype='PCM_24')


def tem_video(caminho):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=codec_type',
                        '-of', 'csv=p=0', caminho], capture_output=True, text=True).stdout
    return 'video' in r


def mono(x):
    return x.mean(0) if x.ndim == 2 else x


def rms_db(x, sr, janela=0.05):
    """Nível em dB a cada `janela` segundos."""
    x = mono(x); h = max(1, int(sr * janela)); n = len(x) // h
    if n == 0: return np.array([-120.0])
    f = x[:n * h].reshape(n, h)
    return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)


def medir(x, sr):
    """Medidas de um editor de som: loudness, pico real, piso de ruído, faixa dinâmica e chiado."""
    import pyloudnorm as pyln
    from scipy.signal import resample_poly
    xs = x.T if x.ndim == 2 else x
    try:
        lufs = pyln.Meter(sr).integrated_loudness(xs)
    except ValueError:
        lufs = float('-inf')
    tp = 20 * np.log10(np.abs(resample_poly(mono(x), 4, 1)).max() + 1e-9)
    db = rms_db(x, sr)
    piso, fala = np.percentile(db, 10), np.percentile(db, 90)
    m = mono(x)
    spec = np.abs(np.fft.rfft(m[:sr * 60] if len(m) > sr * 60 else m))
    fr = np.fft.rfftfreq(min(len(m), sr * 60), 1 / sr)
    total = spec[(fr > 100) & (fr < 12000)].sum() + 1e-9
    sib = spec[(fr > 5000) & (fr < 9000)].sum() / total * 100
    return {'lufs': round(float(lufs), 1), 'pico_dbtp': round(float(tp), 1), 'piso_ruido_db': round(float(piso), 1),
            'relacao_sinal_ruido_db': round(float(fala - piso), 1), 'chiado_s_pct': round(float(sib), 1),
            'zumbido': detectar_zumbido(m, sr)}


def detectar_zumbido(m, sr):
    """Procura zumbido elétrico de 50 ou 60 Hz (e harmônicos)."""
    seg = m[:sr * 30]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    fr = np.fft.rfftfreq(len(seg), 1 / sr)
    for f0 in (50, 60):
        pico = spec[(fr > f0 - 1) & (fr < f0 + 1)].max()
        viz = np.median(spec[(fr > f0 - 15) & (fr < f0 + 15)])
        h2 = spec[(fr > 2 * f0 - 1) & (fr < 2 * f0 + 1)].max() / (np.median(spec[(fr > 2 * f0 - 15) & (fr < 2 * f0 + 15)]) + 1e-9)
        if pico / (viz + 1e-9) > 8 and h2 > 4:
            return f0
    return None


def modelo(padrao):
    achados = sorted(glob.glob(os.path.join(MOD, padrao)))
    if not achados:
        raise SystemExit(f'Modelo não encontrado ({padrao}). Rode ./instalar.sh')
    return achados[0]


def separador():
    import sherpa_onnx as s
    d = modelo('sherpa-onnx-spleeter-2stems*')
    cfg = s.OfflineSourceSeparationConfig(model=s.OfflineSourceSeparationModelConfig(
        spleeter=s.OfflineSourceSeparationSpleeterModelConfig(vocals=glob.glob(d + '/vocals*.onnx')[0],
                                                              accompaniment=glob.glob(d + '/accompaniment*.onnx')[0]),
        num_threads=os.cpu_count() or 4))
    return s.OfflineSourceSeparation(cfg)


def separar(x, sr=44100, bloco=60):
    """Separa voz e fundo (música + efeitos), em blocos para não estourar a memória."""
    sep = separador(); voz, fundo = [], []
    for k in range(0, x.shape[1], sr * bloco):
        out = sep.process(sr, np.ascontiguousarray(x[:, k:k + sr * bloco]))
        voz.append(np.asarray(out.stems[0].data)); fundo.append(np.asarray(out.stems[1].data))
    return np.concatenate(voz, 1), np.concatenate(fundo, 1)


def vad(m16, sr=16000, silencio_min=0.25):
    """Trechos com fala (início, fim) em segundos, usando o Silero VAD."""
    import sherpa_onnx as s
    cfg = s.VadModelConfig(silero_vad=s.SileroVadModelConfig(model=modelo('silero_vad.onnx'),
                                                             min_silence_duration=silencio_min,
                                                             min_speech_duration=0.15, max_speech_duration=25),
                           sample_rate=sr)
    v = s.VoiceActivityDetector(cfg, buffer_size_in_seconds=600); trechos = []
    for k in range(0, len(m16), 512):
        v.accept_waveform(m16[k:k + 512])
        while not v.empty():
            st = v.front.start; trechos.append((st / sr, (st + len(v.front.samples)) / sr)); v.pop()
    v.flush()
    while not v.empty():
        st = v.front.start; trechos.append((st / sr, (st + len(v.front.samples)) / sr)); v.pop()
    return trechos
