#!/usr/bin/env python3
"""Corte vertical (9:16) de um trecho de podcast gravado na horizontal, seguindo o manual de edição.

O que faz:
- remove as pausas e abre o vídeo direto na primeira frase forte;
- no close, recorta o vertical em volta do rosto e alterna zoom (punch-in) a cada corte;
- no plano aberto, alterna entre o rosto de quem fala e o quadro inteiro sobre fundo desfocado;
- legenda em caixa alta, palavra por palavra, com a palavra falada em amarelo;
- cartão de gancho, crédito do podcast e pedido final;
- som: equalização, compressão, emendas suaves e volume final de -14 LUFS.

Uso:
  python3 corte_podcast.py ORIGINAL.mp4 VOZ.wav --modo previa --saida corte.mp4
  python3 corte_podcast.py ORIGINAL.mp4 VOZ.wav --modo final  --saida corte.mp4   # com nitidez por IA
  python3 corte_podcast.py ... --quadros 1,5,9 --saida pasta/
"""
import argparse, os, re, subprocess, sys, tempfile, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..'))
from comum import carregar, salvar_wav

MOD = os.path.join(AQUI, '..', 'modelos')
FONTES = os.path.join(MOD, 'fontes', 'package', 'files')
SW, SH, FPS = 1920, 1080, 25
OW, OH = 1080, 1920

# ---------------- roteiro (tempos do original) ----------------
FRASES = [
    (1.62, 3.48, 'Amyr é sensacional,'),
    (3.95, 7.41, 'baita pessoa, corajoso demais, divertido com seus comentários.'),
    (7.82, 10.29, 'Quero ver quando sai uma navegação com uma'),
    (10.54, 11.60, 'Arca de Noé.'),
    (11.79, 12.69, 'Sucesso!'),
    (12.75, 15.57, 'Eu acho que vai demorar.'),
    (16.75, 19.50, 'Mas eu ando com os caras loucos de vez em quando, que...'),
    (20.27, 21.62, 'chegam a aparecer.'),
    (22.35, 23.28, 'Um conjunto de...'),
    (23.72, 24.88, 'de alucinados.'),
]
# trechos do original que entram no corte (as pausas entre eles saem)
TRECHOS = [(1.60, 3.55), (3.87, 7.50), (7.74, 16.05), (16.67, 19.58), (20.19, 21.70), (22.27, 23.36), (23.64, 25.00)]
TRECHOS = [(round(a * FPS) / FPS, round(b * FPS) / FPS) for a, b in TRECHOS]   # cortes alinhados aos quadros

# enquadramentos, por tempo do original: (início, tipo, foco x, foco y, zoom)
#   'v' = recorte vertical em volta do rosto; 'h' = quadro inteiro sobre fundo desfocado
PLANOS = [
    (0.00, 'v', 0.50, 0.50, 1.00),    # apresentador lendo a mensagem
    (3.87, 'v', 0.50, 0.50, 1.25),    # punch-in no corte
    (7.74, 'v', 0.50, 0.50, 1.00),
    (10.54, 'v', 0.50, 0.48, 1.35),   # "Arca de Noé"
    (13.12, 'v', 0.45, 0.32, 1.00),   # convidado rindo
    (14.45, 'v', 0.45, 0.30, 1.30),
    (16.88, 'v', 0.86, 0.56, 1.40),   # plano aberto: rosto do convidado
    (20.19, 'h', 0.50, 0.50, 1.00),   # mesa inteira, reação dos dois
    (22.27, 'v', 0.86, 0.56, 1.55),
    (23.64, 'v', 0.86, 0.55, 1.85),   # "de alucinados": o mais perto possível
]

AMARELO = (255, 211, 26); PRETO = (17, 17, 17); BRANCO = (255, 255, 255)


def fonte(nome, tam):
    return ImageFont.truetype(os.path.join(FONTES, nome), tam)


def ease(t):
    t = max(0.0, min(1.0, t)); return 1 - (1 - t) ** 3


def ease_back(t):
    t = max(0.0, min(1.0, t)); c = 1.7
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


# ---------------- tempo: original <-> corte ----------------
def para_corte(t):
    acc = 0.0
    for a, b in TRECHOS:
        if a <= t <= b: return acc + (t - a)
        if t < a: return acc
        acc += b - a
    return acc


def para_original(tc):
    acc = 0.0
    for a, b in TRECHOS:
        if tc < acc + (b - a): return a + (tc - acc)
        acc += b - a
    return TRECHOS[-1][1]


DUR = sum(b - a for a, b in TRECHOS)


# ---------------- palavras ----------------
def silabas(p):
    p = re.sub(r'[^\wà-ü]', '', p.lower())
    return max(1, len(re.findall(r'[aeiouyáéíóúâêôãõàü]+', p)))


def tempos_palavras(voz_wav):
    v = carregar(voz_wav, 16000, 1)[0]
    hop = 160; n = len(v) // hop
    db = 20 * np.log10(np.sqrt((v[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    P = []
    for a, b, txt in FRASES:
        ws = txt.split(); sil = np.array([silabas(w) for w in ws], float)
        seg = db[int(a * 100):min(n, int(b * 100))]
        ativo = (seg > np.median(seg) - 12).astype(float)
        cum = np.concatenate([[0], np.cumsum(ativo)]); frac = np.concatenate([[0], np.cumsum(sil)]) / sil.sum()
        t_de = lambda f: a + np.searchsorted(cum, f * cum[-1]) / 100
        for k, w in enumerate(ws):
            P.append({'w': w, 'a': para_corte(t_de(frac[k])), 'b': para_corte(t_de(frac[k + 1]))})
    return P


def blocos(P):
    out, atual = [], []
    for p in P:
        atual.append(p)
        txt = ' '.join(x['w'] for x in atual)
        if len(atual) >= 3 or len(txt) > 14 or re.search(r'[.,?!;]$', p['w']) or p['w'].endswith('...'):
            out.append(atual); atual = []
    if atual: out.append(atual)
    res = []
    for i, bl in enumerate(out):
        ini = bl[0]['a'] - 0.03
        prox = out[i + 1][0]['a'] - 0.03 if i + 1 < len(out) else None
        fim = prox if prox is not None and prox - bl[-1]['b'] < 0.5 else bl[-1]['b'] + 0.3
        res.append({'ini': ini, 'fim': fim, 'pal': bl})
    return res


# ---------------- imagem ----------------
class Ampliador:
    def __init__(self):
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = os.cpu_count() or 4
        self.s = ort.InferenceSession(os.path.join(MOD, 'upscale', 'realesr-general-x4v3.onnx'), o, providers=['CPUExecutionProvider'])

    def __call__(self, im, alvo):
        # entrada limitada a ~480 px de largura para manter ~3 s por quadro
        if im.width > 480: im = im.resize((480, int(im.height * 480 / im.width)), Image.LANCZOS)
        x = np.ascontiguousarray((np.asarray(im, np.float32) / 255).transpose(2, 0, 1)[None])
        y = self.s.run(None, {'entrada': x})[0][0]
        out = Image.fromarray((np.clip(y.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8))
        return out.resize(alvo, Image.LANCZOS)


def plano_em(t_orig):
    return max((p for p in PLANOS if p[0] <= t_orig), key=lambda p: p[0])


def compor(src, t_orig, ampliar):
    _, tipo, fx, fy, z = plano_em(t_orig)
    im = Image.fromarray(src)
    if tipo == 'v':
        h = SH / z; w = h * 9 / 16
        x0 = min(max(fx * SW - w / 2, 0), SW - w); y0 = min(max(fy * SH - 0.42 * h, 0), SH - h)
        crop = im.crop((int(x0), int(y0), int(x0 + w), int(y0 + h)))
        return ampliar(crop, (OW, OH)) if ampliar else crop.resize((OW, OH), Image.LANCZOS)
    # quadro inteiro sobre o próprio vídeo ampliado e desfocado
    fundo = im.resize((int(SW * OH / SH), OH), Image.BILINEAR)
    fundo = fundo.crop(((fundo.width - OW) // 2, 0, (fundo.width - OW) // 2 + OW, OH)).filter(ImageFilter.GaussianBlur(40))
    fundo = Image.eval(fundo, lambda v: int(v * 0.55))
    fh = int(OW * SH / SW)
    frente = ampliar(im, (OW, fh)) if ampliar else im.resize((OW, fh), Image.LANCZOS)
    fundo.paste(frente, (0, int(OH * 0.46 - fh / 2)))
    return fundo


class Grafismos:
    def __init__(self, P):
        self.f_leg = fonte('montserrat-latin-900-normal.woff', 76)
        self.f_card = fonte('montserrat-latin-900-normal.woff', 66)
        self.f_sobre = fonte('montserrat-latin-800-normal.woff', 28)
        self.f_cred = fonte('montserrat-latin-700-normal.woff', 30)
        self.leg = blocos(P)
        self.cta_ini = DUR - 1.6
        self.cache = {}

    def cartao(self, sobre, titulo, cor=AMARELO):
        chave = (sobre, titulo)
        if chave in self.cache: return self.cache[chave]
        d0 = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
        linhas = titulo.split('\n'); larg = 900; pad = 38
        alt = pad + 40 + len(linhas) * 78 + pad
        im = Image.new('RGBA', (larg + 40, alt + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle([20, 28, 20 + larg, 28 + alt], 28, fill=(0, 0, 0, 90))
        d.rounded_rectangle([20, 20, 20 + larg, 20 + alt], 28, fill=cor + (255,))
        w = d.textlength(sobre, font=self.f_sobre)
        d.text((20 + (larg - w) / 2, 20 + pad), sobre, font=self.f_sobre, fill=(60, 50, 0))
        y = 20 + pad + 44
        for ln in linhas:
            w = d.textlength(ln, font=self.f_card); d.text((20 + (larg - w) / 2, y), ln, font=self.f_card, fill=PRETO); y += 78
        self.cache[chave] = im
        return im

    def colar(self, base, im, cx, cy, alfa=1.0, esc=1.0):
        if esc != 1.0: im = im.resize((max(1, int(im.width * esc)), max(1, int(im.height * esc))), Image.BICUBIC)
        if alfa < 1.0:
            im = im.copy(); im.putalpha(im.getchannel('A').point(lambda v: int(v * alfa)))
        base.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

    def legenda(self, base, t):
        bl = next((b for b in self.leg if b['ini'] <= t < b['fim']), None)
        if not bl: return
        d = ImageDraw.Draw(base)
        ativa = max([i for i, p in enumerate(bl['pal']) if p['a'] - 0.03 <= t] or [0])
        ws = [p['w'].upper() for p in bl['pal']]; esp = d.textlength(' ', font=self.f_leg)
        larg = [d.textlength(w, font=self.f_leg) for w in ws]
        x = (OW - (sum(larg) + esp * (len(ws) - 1))) / 2; y = OH * 0.735
        k = ease((t - bl['ini']) / 0.1); esc_y = (1 - k) * 12
        for i, w in enumerate(ws):
            d.text((x, y + esc_y), w, font=self.f_leg, anchor='ls', fill=AMARELO if i == ativa else BRANCO,
                   stroke_width=7, stroke_fill=PRETO)
            x += larg[i] + esp

    def desenhar(self, base, t):
        d = ImageDraw.Draw(base)
        # crédito do podcast
        d.rounded_rectangle([40, 70, 40 + d.textlength('FLOW PODCAST', font=self.f_cred) + 36, 70 + 52], 26, fill=(0, 0, 0, 130))
        d.text((58, 96), 'FLOW PODCAST', font=self.f_cred, fill=BRANCO, anchor='lm')
        # gancho
        if t < 3.2:
            a = ease(t / 0.3) * (1 - ease((t - 2.9) / 0.3)); k = ease_back(t / 0.35)
            self.colar(base, self.cartao('PERGUNTA DO CHAT', 'QUANDO SAI A\nARCA DE NOÉ?'), OW / 2, OH * 0.2, a, 0.85 + 0.15 * k)
        # pedido final
        if t >= self.cta_ini:
            k = ease_back((t - self.cta_ini) / 0.35)
            self.colar(base, self.cartao('GOSTOU?', 'SEGUE PRA MAIS\nCORTES'), OW / 2, OH * 0.2, ease((t - self.cta_ini) / 0.25), 0.85 + 0.15 * k)
        self.legenda(base, t)


# ---------------- vídeo e áudio ----------------
def quadros_original(original):
    cor = 'eq=contrast=1.04:saturation=1.06'
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', original, '-vf', cor, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         stdout=subprocess.PIPE)
    n = SW * SH * 3; i = 0
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield i / FPS, np.frombuffer(b, np.uint8).reshape(SH, SW, 3); i += 1
    p.stdout.close(); p.terminate()


def audio(original, saida_wav):
    from pedalboard import Pedalboard, HighpassFilter, PeakFilter, Compressor
    sr = 48000; x = carregar(original, sr, 2)
    partes = []; fade = int(0.012 * sr)
    for a, b in TRECHOS:
        seg = x[:, int(a * sr):int(b * sr)].copy()
        rampa = np.linspace(0, 1, fade); seg[:, :fade] *= rampa; seg[:, -fade:] *= rampa[::-1]   # emendas sem estalo
        partes.append(seg)
    y = np.concatenate(partes, 1).astype(np.float32)
    y = Pedalboard([HighpassFilter(80), PeakFilter(220, -2.0, 1.0), PeakFilter(3500, 2.5, 0.9),
                    Compressor(threshold_db=-24, ratio=3, attack_ms=6, release_ms=90)])(y, sr)
    rng = np.random.default_rng(1); n = int(0.4 * sr)
    r = rng.standard_normal(n); f = np.fft.rfft(r); fr = np.fft.rfftfreq(n, 1 / sr)
    f *= np.exp(-((fr - 2500) / 2200) ** 2); r = np.fft.irfft(f, n); r = r / np.abs(r).max() * np.sin(np.linspace(0, np.pi, n)) ** 2 * 0.04
    y[:, :n] += r                                           # whoosh discreto na entrada do gancho
    i = int((DUR - 1.6) * sr); y[:, i:i + n] += r[:y.shape[1] - i] * 0.8   # e no pedido final
    import tratar_audio as ta
    ta.SR = sr
    salvar_wav(saida_wav, ta.finalizar(y, -14, -1.5), sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original'); ap.add_argument('voz')
    ap.add_argument('--modo', choices=['previa', 'final'], default='previa'); ap.add_argument('--saida', required=True)
    ap.add_argument('--quadros')
    a = ap.parse_args()
    P = tempos_palavras(a.voz); G = Grafismos(P)
    amp = Ampliador() if a.modo == 'final' else None
    nq = int(round(DUR * FPS))
    alvos = [int(round(para_original(i / FPS) * FPS)) for i in range(nq)]   # para cada quadro do corte, o quadro do original
    if a.quadros:
        os.makedirs(a.saida, exist_ok=True)
        pedidos = sorted(float(x) for x in a.quadros.split(','))
        for t, src in quadros_original(a.original):
            for tc in [q for q in pedidos if int(round(para_original(q) * FPS)) == int(round(t * FPS))]:
                base = compor(src, t, amp).convert('RGBA'); G.desenhar(base, tc)
                base.convert('RGB').save(os.path.join(a.saida, f'c_{tc:05.2f}.png')); pedidos.remove(tc)
            if not pedidos: break
        return
    tmp = tempfile.mkdtemp(); audio(a.original, f'{tmp}/a.wav')
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}',
                            '-r', str(FPS), '-i', '-', '-i', f'{tmp}/a.wav', '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
                            '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '256k',
                            '-shortest', '-movflags', '+faststart', a.saida], stdin=subprocess.PIPE)
    k = 0
    for t, src in quadros_original(a.original):
        while k < nq and alvos[k] == int(round(t * FPS)):
            base = compor(src, t, amp).convert('RGBA'); G.desenhar(base, k / FPS)
            enc.stdin.write(base.convert('RGB').tobytes()); k += 1
            if k % 50 == 0: print(f'quadro {k}/{nq}', flush=True)
        if k >= nq: break
    enc.stdin.close(); enc.wait(); print('pronto', a.saida)


if __name__ == '__main__':
    main()
