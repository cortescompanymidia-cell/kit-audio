#!/usr/bin/env python3
"""Análise de edição de um vídeo de referência: ritmo de cortes, gancho, visual e áudio.

Uso:  python3 analisar_video.py VIDEO.mp4 [--saida pasta]
Gera na pasta:
  resumo.md         números de ritmo, gancho e áudio
  cortes.txt        momentos de mudança visual (cortes e transições agrupados)
  folha_NN.png      1 quadro a cada 2 s, com o tempo escrito (para ler legenda, layout e grafismo)
  gancho.png        1 quadro a cada 0,5 s nos 6 primeiros segundos
"""
import argparse, os, re, subprocess, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from comum import carregar, medir, rms_db

FONTE = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def info(v):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=width,height,r_frame_rate:format=duration', '-of', 'default=nw=1', v],
                       capture_output=True, text=True).stdout
    d = dict(l.split('=') for l in r.strip().splitlines())
    n, m = d['r_frame_rate'].split('/')
    return int(d['width']), int(d['height']), float(n) / float(m), float(d['duration'])


def mudancas(v, limiar):
    e = subprocess.run(['ffmpeg', '-hide_banner', '-i', v, '-vf', f"select='gt(scene,{limiar})',showinfo", '-an', '-f', 'null', '-'],
                       capture_output=True, text=True).stderr
    return [float(t) for t in re.findall(r'pts_time:([\d.]+)', e)]


def agrupar(ts, janela=0.4):
    """Mudanças muito próximas são uma transição animada só."""
    grupos = []
    for t in ts:
        if grupos and t - grupos[-1][-1] <= janela: grupos[-1].append(t)
        else: grupos.append([t])
    return grupos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video'); ap.add_argument('--saida'); ap.add_argument('--limiar', type=float, default=0.08)
    a = ap.parse_args()
    out = a.saida or os.path.splitext(a.video)[0] + '_analise'
    os.makedirs(out, exist_ok=True)
    w, h, fps, dur = info(a.video)
    vertical = h > w
    esc = '216:384' if vertical else '320:-2'
    cols, linhas = (10, 3) if vertical else (6, 6)
    print('quadros…', flush=True)
    txt = f"drawtext=fontfile={FONTE}:text='%{{pts\\:hms}}':x=4:y=4:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.7"
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', a.video, '-vf', f'fps=1/2,scale={esc},{txt},tile={cols}x{linhas}',
                    '-fps_mode', 'vfr', f'{out}/folha_%02d.png'], check=True)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-t', '6', '-i', a.video, '-vf', f'fps=2,scale={esc},{txt},tile=12x1',
                    '-frames:v', '1', f'{out}/gancho.png'], check=True)
    print('cortes…', flush=True)
    g = agrupar(mudancas(a.video, a.limiar))
    ev = [x[0] for x in g]
    anim = [x[0] for x in g if len(x) >= 3]
    open(f'{out}/cortes.txt', 'w').write('\n'.join(f'{t:.2f}{"  (transição animada)" if t in anim else ""}' for t in ev) + '\n')
    print('áudio…', flush=True)
    sr = 16000; x = carregar(a.video, sr, 1)[0]
    m = medir(x, sr); db = rms_db(x, sr)
    quieto = db < np.percentile(db, 50) - 18
    pausas, c = [], 0
    for q in quieto:
        if q: c += 1
        elif c: pausas.append(c * 0.05); c = 0
    pausas = np.array(pausas or [0])
    ini = [t for t in ev if t < 10]; dep = [t for t in ev if t >= 10]
    primeira = next((t for t in ev if t > 0.05), None)
    R = [f'# Análise · {os.path.basename(a.video)}', '',
         f'- **Formato:** {w}×{h} ({"vertical" if vertical else "horizontal"}), {fps:.2f} qps, {dur:.1f} s',
         f'- **Mudanças visuais:** {len(ev)} (uma a cada {dur / (len(ev) + 1):.1f} s), {len(anim)} delas transições animadas',
         f'- **Primeira mudança:** {primeira:.1f} s' if primeira is not None else '- **Primeira mudança:** nenhuma',
         f'- **Primeiros 10 s:** {len(ini)} mudanças · **depois:** uma a cada {(dur - 10) / max(1, len(dep)):.1f} s',
         f'- **Silêncio:** {quieto.mean() * 100:.1f}% do vídeo · maior pausa {pausas.max():.2f} s · pausas acima de 0,25 s: {(pausas > 0.25).sum()}',
         f'- **Áudio:** {m["lufs"]} LUFS · pico {m["pico_dbtp"]} dBTP · piso {m["piso_ruido_db"]} dB'
         f'{" (há trilha contínua por baixo)" if m["piso_ruido_db"] > -38 else ""}', '',
         'Abra `gancho.png` e as `folha_NN.png` para ler legendas, layout, grafismos e transições.',
         'Para o som (fala, efeitos, música), rode `../mapa_sonoro.py` no mesmo vídeo.']
    open(f'{out}/resumo.md', 'w').write('\n'.join(R) + '\n')
    print('\n'.join(R))


if __name__ == '__main__':
    main()
