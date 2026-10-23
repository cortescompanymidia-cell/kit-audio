#!/usr/bin/env python3
"""Mapa sonoro: "escuta" um vídeo ou áudio e descreve, com tempo, a fala, a música e os efeitos sonoros.

Uso:  python3 mapa_sonoro.py VIDEO.mp4 [--idioma pt] [--whisper medium] [--saida pasta]
Gera: mapa.md (leitura humana), mapa.json (dados) e as faixas separadas voz.wav / fundo.wav.
"""
import argparse, json, os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum import carregar, salvar_wav, tem_video, mono, rms_db, medir, modelo, separar, vad

# Rótulos genéricos do AudioSet que não ajudam a descrever um efeito.
GENERICOS = {'Music', 'Musical instrument', 'Speech', 'Silence', 'Inside, small room', 'Inside, large room or hall',
             'Narration, monologue', 'Male speech, man speaking', 'Female speech, woman speaking', 'Conversation',
             'Speech synthesizer', 'Sound effect', 'Noise', 'Background music', 'Theme music', 'Video game music',
             'Soundtrack music', 'Outside, urban or manmade', 'Outside, rural or natural', 'Child speech, kid speaking'}
PT = {
    'Whoosh, swoosh, swish': 'whoosh (passagem de ar)', 'Boom': 'impacto grave (boom)', 'Thump, thud': 'batida surda',
    'Bang': 'estouro', 'Explosion': 'explosão', 'Slam': 'batida forte', 'Smash, crash': 'quebra / crash',
    'Breaking': 'quebra', 'Shatter': 'estilhaço', 'Glass': 'vidro', 'Camera': 'câmera', 'Click': 'clique',
    'Mouse click': 'clique de mouse', 'Typing': 'digitação', 'Computer keyboard': 'teclado',
    'Beep, bleep': 'bipe', 'Sine wave': 'tom puro / bipe', 'Ding': 'ding', 'Chime': 'sino', 'Bell': 'sino',
    'Ringtone': 'toque de celular', 'Alarm': 'alarme', 'Buzzer': 'buzina elétrica', 'Zipper (clothing)': 'zíper',
    'Applause': 'aplausos', 'Clapping': 'palmas', 'Cheering': 'torcida', 'Laughter': 'risadas', 'Crowd': 'multidão',
    'Gasp': 'susto (respiração)', 'Sigh': 'suspiro', 'Breathing': 'respiração', 'Footsteps': 'passos',
    'Drum': 'tambor', 'Bass drum': 'bumbo', 'Snare drum': 'caixa', 'Cymbal': 'prato', 'Hi-hat': 'chimbal',
    'Drum kit': 'bateria', 'Drum machine': 'bateria eletrônica', 'Clapping ': 'palmas', 'Scratching (performance technique)': 'scratch de DJ',
    'Synthesizer': 'sintetizador', 'Electronic music': 'música eletrônica', 'Hip hop music': 'hip hop',
    'Techno': 'techno', 'House music': 'house', 'Dubstep': 'dubstep', 'Drum and bass': 'drum and bass',
    'Electronic dance music': 'música eletrônica de pista', 'Ambient music': 'música ambiente', 'Pop music': 'pop',
    'Rock music': 'rock', 'Orchestra': 'orquestra', 'Piano': 'piano', 'Guitar': 'guitarra/violão', 'Bass guitar': 'baixo',
    'Funk': 'funk', 'Trap music': 'trap', 'Cinematic': 'cinematográfica', 'Happy music': 'música alegre',
    'Sad music': 'música triste', 'Tender music': 'música suave', 'Exciting music': 'música empolgante',
    'Angry music': 'música tensa', 'Scary music': 'música de suspense', 'Funny music': 'música engraçada',
    'Wind': 'vento', 'Rain': 'chuva', 'Thunder': 'trovão', 'Water': 'água', 'Bird': 'pássaro', 'Dog': 'cachorro',
    'Vehicle': 'veículo (num corte, costuma ser whoosh)', 'Car': 'carro', 'Engine': 'motor', 'Traffic noise, roadway noise': 'trânsito',
    'Reverberation': 'reverberação', 'Echo': 'eco', 'Static': 'estática', 'White noise': 'ruído branco',
    'Pink noise': 'ruído rosa', 'Hum': 'zumbido', 'Mains hum': 'zumbido elétrico', 'Effects unit': 'efeito de áudio',
    'Chirp, tweet': 'piado', 'Tick': 'tique', 'Tick-tock': 'tique-taque', 'Clock': 'relógio', 'Cash register': 'caixa registradora',
    'Coin (dropping)': 'moeda', 'Pop': 'pop (estalo)', 'Swoosh': 'whoosh', 'Rumble': 'ronco grave', 'Sizzle': 'chiado',
    'Crackle': 'estalos', 'Sonar': 'ping eletrônico / bipe sintético', 'Heart murmur': 'pulso grave (provável sub drop)', 'Throbbing': 'pulsação grave', 'Whack, thwack': 'pancada', 'Chirp tone': 'tom ascendente (chirp)',
    'Cricket': 'grilo', 'Insect': 'inseto', 'Crunch': 'estalo', 'Rattle': 'chocalho', 'Harmonic': 'tom harmônico',
    'Heart sounds, heartbeat': 'pulso grave (batimento / sub drop)', 'Sidetone': 'tom de linha', 'Telephone': 'telefone', 'Rustle': 'farfalhar',
    'Squeak': 'rangido', 'Creak': 'rangido', 'Pour': 'líquido sendo servido', 'Writing': 'escrita', 'Tap': 'toque leve', 'Fire': 'fogo', 'Gunshot, gunfire': 'tiro', 'Robotic': 'som robótico',
}
pt = lambda n: PT.get(n, n)


def tagger(top_k=8):
    import sherpa_onnx as s
    d = modelo('sherpa-onnx-ced-*audio-tagging*')
    cfg = s.AudioTaggingConfig(model=s.AudioTaggingModelConfig(ced=d + '/model.int8.onnx', num_threads=os.cpu_count() or 4),
                               labels=d + '/class_labels_indices.csv', top_k=top_k)
    return s.AudioTagging(cfg)


def rotulos_musicais():
    """Nomes da árvore 'Music' do AudioSet (índices 137 a 282): instrumentos, gêneros e climas musicais."""
    import csv
    d = modelo('sherpa-onnx-ced-*audio-tagging*')
    return {r[2] for r in csv.reader(open(d + '/class_labels_indices.csv')) if r[0].isdigit() and 137 <= int(r[0]) <= 282}


def rotular(tg, seg16, limiar=0.12):
    st = tg.create_stream(); st.accept_waveform(16000, np.ascontiguousarray(seg16, np.float32))
    return [(e.name, float(e.prob)) for e in tg.compute(st) if e.prob >= limiar]


def cortes_video(caminho, limiar=0.25):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', caminho, '-vf', f"select='gt(scene,{limiar})',showinfo",
                        '-an', '-f', 'null', '-'], capture_output=True, text=True).stderr
    import re
    ts = [float(t) for t in re.findall(r'pts_time:([\d.]+)', r)]
    out = []
    for t in ts:
        if not out or t - out[-1] > 0.3: out.append(t)
    return out


def transcrever(m16, trechos, idioma, tamanho):
    import sherpa_onnx as s
    d = modelo(f'sherpa-onnx-whisper-{tamanho}')
    pick = lambda p: (sorted(__import__('glob').glob(f'{d}/*{p}.int8.onnx')) or sorted(__import__('glob').glob(f'{d}/*{p}.onnx')))[0]
    rec = s.OfflineRecognizer.from_whisper(encoder=pick('encoder'), decoder=pick('decoder'),
                                           tokens=__import__('glob').glob(f'{d}/*tokens.txt')[0],
                                           language=idioma, task='transcribe', num_threads=os.cpu_count() or 4)
    falas = []
    # junta trechos próximos (até 0,6 s) para frases mais completas, sem passar de 25 s
    grupos = []
    for a, b in trechos:
        if grupos and a - grupos[-1][1] < 0.6 and b - grupos[-1][0] < 25: grupos[-1][1] = b
        else: grupos.append([a, b])
    for a, b in grupos:
        st = rec.create_stream()
        st.accept_waveform(16000, m16[int(max(0, a - 0.15) * 16000):int((b + 0.15) * 16000)])
        rec.decode_stream(st)
        t = st.result.text.strip()
        if t: falas.append({'inicio': round(a, 2), 'fim': round(b, 2), 'texto': t})
    return falas


def mmss(t):
    return f'{int(t // 60)}:{t % 60:04.1f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arquivo'); ap.add_argument('--idioma', default='pt', help='pt, en, es… (idioma da fala)')
    ap.add_argument('--whisper', default='medium', choices=['small', 'medium'])
    ap.add_argument('--saida', default=None); ap.add_argument('--sem-fala', action='store_true', help='pula a transcrição')
    a = ap.parse_args()
    import librosa
    base = os.path.splitext(os.path.basename(a.arquivo))[0]
    out = a.saida or os.path.join(os.path.dirname(os.path.abspath(a.arquivo)), base + '_mapa')
    os.makedirs(out, exist_ok=True)
    SR = 44100
    x = carregar(a.arquivo, SR, 2); dur = x.shape[1] / SR
    print(f'[1/6] {base}: {dur:.1f}s, separando voz e fundo…', flush=True)
    voz, fundo = separar(x, SR)
    salvar_wav(f'{out}/voz.wav', voz, SR); salvar_wav(f'{out}/fundo.wav', fundo, SR)
    to16 = lambda y: librosa.resample(mono(y).astype(np.float32), orig_sr=SR, target_sr=16000)
    voz16, fundo16, mix16 = to16(voz), to16(fundo), to16(x)

    print('[2/6] detectando fala…', flush=True)
    trechos = vad(voz16)
    fala_total = sum(b - a_ for a_, b in trechos)

    print('[3/6] lendo a música…', flush=True)
    fm = mono(fundo)
    tempo, beats = librosa.beat.beat_track(y=fm, sr=SR, units='time')
    bpm = float(np.atleast_1d(tempo)[0])
    while bpm > 180: bpm /= 2  # o detector às vezes conta o dobro
    while 0 < bpm < 70: bpm *= 2
    en = rms_db(fundo, SR, 0.5)
    eventos_musica = []
    for i in range(4, len(en)):
        antes = np.mean(en[i - 4:i])
        if en[i] - antes <= -8 and (not eventos_musica or i * 0.5 - eventos_musica[-1]['t'] > 2):
            eventos_musica.append({'t': i * 0.5, 'tipo': 'queda', 'db': round(float(en[i] - antes), 1)})
        elif en[i] - antes >= 8 and (not eventos_musica or i * 0.5 - eventos_musica[-1]['t'] > 2):
            eventos_musica.append({'t': i * 0.5, 'tipo': 'subida', 'db': round(float(en[i] - antes), 1)})

    print('[4/6] identificando efeitos e ambientes…', flush=True)
    tg = tagger()
    oenv = librosa.onset.onset_strength(y=fundo16, sr=16000)
    ot = librosa.times_like(oenv, sr=16000)
    picos = librosa.util.peak_pick(oenv, pre_max=8, post_max=8, pre_avg=20, post_avg=20, delta=np.percentile(oenv, 90) * 0.5, wait=8)
    forca_ref = np.median(oenv) + 1e-9
    MUSICAIS = rotulos_musicais()
    efeitos, golpes = [], []  # golpes: impactos fortes que são da própria música (bumbo, caixa, drop)
    for p in picos:
        t = float(ot[p]); forca = float(oenv[p] / forca_ref)
        if forca < 4: continue
        seg = fundo16[int(max(0, t - 0.15) * 16000):int((t + 0.85) * 16000)]
        if len(seg) < 8000: continue
        rots = [(n, pr) for n, pr in rotular(tg, seg) if n not in GENERICOS]
        sfx = [(n, pr) for n, pr in rots if n not in MUSICAIS]
        if sfx:
            efeitos.append({'t': round(t, 2), 'forca': round(forca, 1), 'sons': [(pt(n), n, round(pr, 2)) for n, pr in sfx[:3]]})
        elif forca >= 6:
            golpes.append({'t': round(t, 2), 'forca': round(forca, 1)})
    # ambiente / estilo musical em janelas de 3 s
    ambientes = []
    for k in np.arange(0, max(dur - 1, 0.1), 3.0):
        seg = fundo16[int(k * 16000):int((k + 3) * 16000)]
        if len(seg) < 16000: break
        rots = rotular(tg, seg, 0.1)
        musica = dict(rots).get('Music', 0)
        resto = [(pt(n), round(pr, 2)) for n, pr in rots if n not in GENERICOS][:3]
        ambientes.append({'t': float(k), 'musica': round(musica, 2), 'descricao': resto})

    cortes = []
    if tem_video(a.arquivo):
        print('[5/6] detectando cortes do vídeo…', flush=True)
        for c in cortes_video(a.arquivo):
            db_ = float(np.min(np.abs(beats - c))) if len(beats) else 9
            ef = [e for e in efeitos if abs(e['t'] - c) <= 0.12]
            gl = any(abs(g['t'] - c) <= 0.12 for g in golpes)
            cortes.append({'t': round(c, 2), 'na_batida': bool(db_ <= 0.07), 'no_golpe': gl,
                           'efeito': ef[0]['sons'][0][0] if ef else None})

    falas = []
    if not a.sem_fala and trechos:
        print(f'[6/6] transcrevendo fala (whisper {a.whisper})…', flush=True)
        falas = transcrever(mix16, trechos, a.idioma, a.whisper)

    med = medir(x, SR)
    # quanto a música fica abaixo da voz quando há fala
    def nivel(sig16, ivs):
        if not ivs: return None
        v = np.concatenate([sig16[int(s_ * 16000):int(e * 16000)] for s_, e in ivs])
        return 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-9)
    nv, nf = nivel(voz16, trechos), nivel(fundo16, trechos)
    sem = []; prev = 0
    for s_, e in trechos:
        if s_ - prev > 0.5: sem.append((prev, s_))
        prev = e
    if dur - prev > 0.5: sem.append((prev, dur))
    nf_sem = nivel(fundo16, sem)
    mix_info = {'voz_sobre_fundo_db': round(float(nv - nf), 1) if nv is not None and nf is not None else None,
                'ducking_db': round(float(nf_sem - nf), 1) if nf_sem is not None and nf is not None else None}

    dados = {'arquivo': os.path.basename(a.arquivo), 'duracao_s': round(dur, 2), 'medidas': med, 'mixagem': mix_info,
             'musica': {'bpm': round(bpm, 1), 'batida_s': round(60 / bpm, 3) if bpm else None, 'eventos': eventos_musica,
                        'energia_db_meio_segundo': [round(float(v), 1) for v in en]},
             'fala': {'trechos': [[round(s_, 2), round(e, 2)] for s_, e in trechos], 'percentual': round(fala_total / dur * 100, 1),
                      'transcricao': falas},
             'efeitos': efeitos, 'golpes_musica': golpes, 'ambientes': ambientes, 'cortes': cortes}
    json.dump(dados, open(f'{out}/mapa.json', 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    # ---------- relatório legível ----------
    L = [f'# Mapa sonoro · {os.path.basename(a.arquivo)}', '',
         f'**Duração:** {mmss(dur)} · **Fala:** {dados["fala"]["percentual"]}% do tempo · **Música:** ~{bpm:.0f} BPM (batida a cada {60 / bpm:.2f} s)', '',
         '## Medidas técnicas', '',
         f'| Loudness | Pico real | Piso de ruído | Sinal/ruído | Chiado (5–9 kHz) | Zumbido | Voz acima do fundo | Ducking |',
         '|---|---|---|---|---|---|---|---|',
         f'| {med["lufs"]} LUFS | {med["pico_dbtp"]} dBTP | {med["piso_ruido_db"]} dB | {med["relacao_sinal_ruido_db"]} dB | {med["chiado_s_pct"]}% | '
         f'{str(med["zumbido"]) + " Hz" if med["zumbido"] else "não"} | {mix_info["voz_sobre_fundo_db"]} dB | {mix_info["ducking_db"]} dB |', '']
    if cortes:
        nb = sum(c['na_batida'] for c in cortes); ne = sum(1 for c in cortes if c['efeito'])
        ng = sum(1 for c in cortes if c['no_golpe'])
        L += [f'**Cortes:** {len(cortes)} · {nb} na batida (±70 ms) · {ng} num golpe forte da música · {ne} com efeito sonoro no instante do corte', '']
    L += [f'**Golpes fortes da música:** {len(golpes)} (impactos de bumbo, caixa ou drop; ficam no mapa.json)', '']
    L += ['## Linha do tempo', '', '| Tempo | Camada | O que acontece |', '|---|---|---|']
    linhas = []
    for f in falas: linhas.append((f['inicio'], 'Fala', f'"{f["texto"]}"'))
    for e in efeitos:
        linhas.append((e['t'], 'Efeito', ', '.join(f'{p} ({int(pr * 100)}%)' for p, _, pr in e['sons']) + f' · força {e["forca"]}x'))
    for m in eventos_musica: linhas.append((m['t'], 'Música', f'{m["tipo"]} de {abs(m["db"])} dB'))
    ult = None
    for amb in ambientes:
        d = ', '.join(n for n, _ in amb['descricao']) or ('música' if amb['musica'] > 0.3 else 'sem destaque')
        if d != ult: linhas.append((amb['t'], 'Fundo', d)); ult = d
    for c in cortes:
        extra = ' · na batida' if c['na_batida'] else ''
        extra += ' · no golpe da música' if c['no_golpe'] else ''
        extra += f' · com {c["efeito"]}' if c['efeito'] else ''
        linhas.append((c['t'], 'Corte', 'corte de imagem' + extra))
    ordem = {'Fundo': 0, 'Música': 1, 'Corte': 2, 'Efeito': 3, 'Fala': 4}
    for t, cam, txt in sorted(linhas, key=lambda r: (r[0], ordem[r[1]])):
        L.append(f'| {mmss(t)} | {cam} | {txt.replace("|", "/")} |')
    L += ['', '_Os nomes dos sons vêm de um classificador automático (AudioSet). A porcentagem é a confiança do modelo. '
              'Os arquivos voz.wav e fundo.wav nesta pasta permitem ouvir cada camada separada._']
    open(f'{out}/mapa.md', 'w').write('\n'.join(L) + '\n')
    print(f'Pronto: {out}/mapa.md')


if __name__ == '__main__':
    main()
