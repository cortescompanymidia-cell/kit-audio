#!/usr/bin/env python3
"""Reedição de um vídeo vertical (nutrigenômica) aplicando o manual de edição.

Uso:
  python3 editar_nutrigenomica.py ORIGINAL.mp4 VOZ.wav FUNDO.wav --modo previa --saida previa.mp4
  python3 editar_nutrigenomica.py ORIGINAL.mp4 VOZ.wav FUNDO.wav --modo final  --saida final_4k.mp4
  python3 editar_nutrigenomica.py ... --quadros 1,8,23,36 --saida pasta/   # só alguns quadros, para conferir

VOZ.wav e FUNDO.wav são as faixas separadas pelo mapa_sonoro.py.
No modo final, cada quadro é ampliado com IA (Real-ESRGAN) para 2160×3840.
"""
import argparse, math, os, re, subprocess, sys, tempfile, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..'))
from comum import carregar, salvar_wav

MOD = os.path.join(AQUI, '..', 'modelos')
FONTES = os.path.join(MOD, 'fontes', 'package', 'files')
SW, SH, FPS = 720, 1280, 30

# ---------------- roteiro corrigido (trechos de fala detectados) ----------------
FRASES = [
    (0.11, 7.57, 'Você sabia que se você tem uma predisposição genética a desenvolver algum tipo de doença hereditária, a alimentação consegue ajudar a evitá-la?'),
    (7.72, 22.29, 'Hoje a gente já sabe que o estilo de vida e principalmente a alimentação têm um papel muito importante na modulação da expressão gênica. '
                  'E é isso que a gente chama de nutrigenômica. Isso acontece por conta dos mecanismos epigenéticos, que não alteram o DNA em si, '
                  'mas sim a forma como os genes são expressos,'),
    (22.44, 23.28, 'se você ativa'),
    (23.43, 24.85, 'ou desativa cada um deles.'),
    (25.10, 26.80, 'E os nutrientes, os compostos bioativos,'),
    (26.95, 34.74, 'influenciam diretamente nesse processo. Para isso, você precisa ter uma alimentação equilibrada. Então, uma dieta rica em verduras, legumes e frutas.'),
    (35.08, 41.49, 'Por outro lado, se a gente tem uma dieta que é rica em ultraprocessados, açúcares e gorduras, que causam uma inflamação,'),
    (41.55, 43.31, 'eles podem ativar os genes'),
    (43.37, 48.75, 'que no futuro podem desenvolver essa doença. Às vezes a gente não vê a importância das nossas escolhas de hoje em dia,'),
    (48.97, 52.29, 'mas são elas que vão impactar diretamente no nosso futuro e na nossa saúde.'),
]
SILABAS_ESPECIAIS = {'dna': 3}

# ---------------- cores e fontes ----------------
CREME = (251, 246, 238); TINTA = (43, 33, 28); VERDE = (46, 107, 78); VERDE_ON = (46, 158, 98)
VERMELHO = (200, 85, 61); CINZA = (185, 178, 168)


def fonte(nome, tam):
    return ImageFont.truetype(os.path.join(FONTES, nome), tam)


def ease(t):
    t = max(0.0, min(1.0, t)); return 1 - (1 - t) ** 3


def ease_back(t):
    t = max(0.0, min(1.0, t)); c = 1.7
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


# ---------------- tempo de cada palavra ----------------
def silabas(p):
    p = re.sub(r'[^\wà-ü]', '', p.lower())
    if p in SILABAS_ESPECIAIS: return SILABAS_ESPECIAIS[p]
    return max(1, len(re.findall(r'[aeiouáéíóúâêôãõàü]+', p)))


def tempos_palavras(voz_wav):
    """Distribui as palavras de cada trecho pelo tempo em que há voz de fato (energia), proporcional às sílabas."""
    v = carregar(voz_wav, 16000, 1)[0]
    hop = 160; n = len(v) // hop
    db = 20 * np.log10(np.sqrt((v[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    palavras = []
    for a, b, txt in FRASES:
        ws = txt.split(); sil = np.array([silabas(w) for w in ws], float)
        i0, i1 = int(a * 100), min(n, int(b * 100))
        seg = db[i0:i1]
        voz_ativa = (seg > np.median(seg) - 12).astype(float)
        cum = np.concatenate([[0], np.cumsum(voz_ativa)]); total = cum[-1]
        frac = np.concatenate([[0], np.cumsum(sil)]) / sil.sum()
        t_de = lambda f: a + np.searchsorted(cum, f * total) / 100
        for k, w in enumerate(ws):
            palavras.append({'w': w, 'a': round(t_de(frac[k]), 3), 'b': round(t_de(frac[k + 1]), 3)})
    return palavras


def blocos_legenda(palavras):
    blocos, atual = [], []
    for p in palavras:
        atual.append(p)
        texto = ' '.join(x['w'] for x in atual)
        if len(atual) >= 3 or len(texto) > 15 or re.search(r'[.,?!;]$', p['w']):
            blocos.append(atual); atual = []
    if atual: blocos.append(atual)
    out = []
    for i, bl in enumerate(blocos):
        ini = bl[0]['a'] - 0.04
        prox = blocos[i + 1][0]['a'] - 0.04 if i + 1 < len(blocos) else None
        fim = prox if prox is not None and prox - bl[-1]['b'] < 0.6 else bl[-1]['b'] + 0.25
        out.append({'ini': ini, 'fim': fim, 'pal': bl})
    return out


def achar(palavras, alvo, depois=0.0):
    alvo = alvo.lower()
    for p in palavras:
        if p['a'] >= depois and re.sub(r'[^\wà-ü]', '', p['w'].lower()).startswith(alvo):
            return p['a']
    raise ValueError(alvo)


# ---------------- enquadramentos ----------------
ROSTO = {'L2': (0.21, 0.40), 'R2': (0.80, 0.40), 'L1': (0.37, 0.40), 'R1': (0.50, 0.42), 'C2': (0.50, 0.40)}


def janela(rosto, z, limite=0.78):
    """Recorte normalizado (x0, y0, w, h) com o rosto a 36% da altura e a base acima da legenda antiga."""
    fx, fy = ROSTO[rosto]; w = h = 1 / z
    x0 = min(max(fx - 0.5 * w, 0), 1 - w)
    y0 = min(max(fy - 0.36 * h, 0), max(0.0, limite - h))
    return x0, y0, w, h


def plano_de_cortes(P):
    t = lambda alvo, depois=0.0: achar(P, alvo, depois)
    return [
        (0.0, 'C2', 1.30),                       # abertura: as duas, com o título
        (t('predisposição'), 'L2', 1.85),        # punch-in na fala
        (t('alimentação'), 'L2', 1.45),
        (t('hoje'), 'L2', 1.70),
        (t('modulação'), 'L2', 1.45),
        (t('isso', 12), 'C2', 1.30),             # aberto para o texto "nutrigenômica" do original
        (16.6, 'L1', 1.30),                      # depois da transição do original
        (t('não', 18), 'L1', 1.70),
        (22.44, 'L1', 1.30),                     # "se você ativa ou desativa": espaço para o interruptor
        (25.10, 'L1', 1.70),
        (28.50, 'R1', 1.30),                     # corte do original para a segunda apresentadora
        (t('você', 29), 'R1', 1.70),
        (t('então', 31), 'R1', 1.163, 0.86),     # aberto para os adesivos do original (verduras, legumes, frutas)
        (35.08, 'R1', 1.75),                     # "Por outro lado"
        (t('ultraprocessados'), 'R1', 1.32),     # espaço para a lista do que inflama
        (41.55, 'R1', 1.75),                     # "eles podem ativar os genes"
        (43.37, 'R1', 1.30),                     # transição de luz do original
        (44.87, 'R2', 1.80),                     # as duas de novo; fala a da direita
        (t('vezes', 45), 'C2', 1.30),
        (48.97, 'L2', 1.80),                     # "mas são elas…"
        (51.25, 'C2', 1.30),                     # final aberto, com o pedido
    ]


def recorte_no_tempo(cortes, t):
    i = max(k for k, c in enumerate(cortes) if c[0] <= t)
    c = cortes[i]; ini = c[0]; fim = cortes[i + 1][0] if i + 1 < len(cortes) else 52.5
    lim = c[3] if len(c) > 3 else 0.78
    z = c[2] * (1 + 0.035 * (t - ini) / max(0.5, fim - ini))   # zoom lento dentro do plano
    if len(c) > 3: z = c[2]                                       # plano dos adesivos fica parado
    return janela(c[1], z, lim)


# ---------------- grafismos ----------------
class Grafismos:
    def __init__(self, W, P):
        self.W = W; self.H = int(W * 16 / 9); self.s = W / 1080; self.P = P
        s = self.s
        self.f_leg = fonte('montserrat-latin-800-normal.woff', int(60 * s))
        self.f_sobre = fonte('montserrat-latin-700-normal.woff', int(27 * s))
        self.f_tit = fonte('dm-serif-display-latin-400-normal.woff', int(78 * s))
        self.f_corpo = fonte('montserrat-latin-500-normal.woff', int(34 * s))
        self.f_tag = fonte('montserrat-latin-800-normal.woff', int(56 * s))
        self.leg = blocos_legenda(P)
        a = lambda alvo, depois=0.0: achar(P, alvo, depois)
        self.ev = {
            'gancho': (0.12, 3.5),
            'epi': (a('epigenéticos'), a('epigenéticos') + 3.4),
            'interruptor': (22.30, 25.0, a('ativa', 22), a('desativa', 23)),
            'ruins': (a('ultraprocessados'), 41.45, [a('ultraprocessados'), a('açúcares'), a('gorduras')]),
            'gene_ruim': (41.50, 43.35, a('ativar', 41)),
            'cta': (49.7, 99),
        }
        self.cache = {}

    # --- utilidades de desenho ---
    def texto_espacado(self, d, xy, txt, f, cor, esp):
        x, y = xy
        for ch in txt:
            d.text((x, y), ch, font=f, fill=cor); x += d.textlength(ch, font=f) + esp

    def largura_espacada(self, d, txt, f, esp):
        return sum(d.textlength(ch, font=f) + esp for ch in txt) - esp

    def cartao(self, sobre, titulo, corpo=None, cor_sobre=VERDE):
        chave = ('cartao', sobre, titulo, corpo)
        if chave in self.cache: return self.cache[chave]
        s = self.s; pad = int(34 * s); larg = int(860 * s)
        tmp = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
        linhas = titulo.split('\n')
        alt = pad + int(36 * s) + len(linhas) * int(86 * s) + (int(52 * s) if corpo else 0) + pad
        im = Image.new('RGBA', (larg + 60, alt + 60), (0, 0, 0, 0))
        sombra = Image.new('RGBA', im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sombra).rounded_rectangle([30, 36, 30 + larg, 36 + alt], int(30 * s), fill=(60, 40, 25, 70))
        im = Image.alpha_composite(im, sombra.filter(ImageFilter.GaussianBlur(14 * s)))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([30, 30, 30 + larg, 30 + alt], int(30 * s), fill=CREME + (242,))
        y = 30 + pad
        w = self.largura_espacada(tmp, sobre, self.f_sobre, 4 * s)
        self.texto_espacado(d, (30 + (larg - w) / 2, y), sobre, self.f_sobre, cor_sobre, 4 * s)
        y += int(42 * s)
        for ln in linhas:
            w = d.textlength(ln, font=self.f_tit)
            d.text((30 + (larg - w) / 2, y), ln, font=self.f_tit, fill=TINTA); y += int(86 * s)
        if corpo:
            w = d.textlength(corpo, font=self.f_corpo)
            d.text((30 + (larg - w) / 2, y + int(4 * s)), corpo, font=self.f_corpo, fill=(95, 82, 72))
        self.cache[chave] = im
        return im

    def etiqueta(self, txt, cor, ang):
        chave = ('tag', txt, cor, ang)
        if chave in self.cache: return self.cache[chave]
        s = self.s; tmp = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
        w = int(tmp.textlength(txt, font=self.f_tag)); h = int(70 * s); pad = int(30 * s)
        im = Image.new('RGBA', (w + 2 * pad + 40, h + 40), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([20, 24, 20 + w + 2 * pad, 24 + h], int(18 * s), fill=(0, 0, 0, 60))
        d.rounded_rectangle([20, 20, 20 + w + 2 * pad, 20 + h], int(18 * s), fill=cor + (255,))
        d.text((20 + pad, 20 + h / 2), txt, font=self.f_tag, fill=(255, 255, 255), anchor='lm')
        im = im.rotate(ang, resample=Image.BICUBIC, expand=True)
        self.cache[chave] = im
        return im

    def interruptor(self, ligado, cor_on, rotulo):
        s = self.s
        im = Image.new('RGBA', (int(520 * s), int(150 * s)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, im.width - 1, im.height - 1], int(40 * s), fill=CREME + (242,))
        d.text((int(40 * s), im.height / 2), rotulo, font=self.f_tag, fill=TINTA, anchor='lm')
        x0, y0, x1, y1 = int(300 * s), int(35 * s), int(480 * s), int(115 * s)
        cor = tuple(int(CINZA[i] + (cor_on[i] - CINZA[i]) * ligado) for i in range(3))
        d.rounded_rectangle([x0, y0, x1, y1], (y1 - y0) // 2, fill=cor)
        r = (y1 - y0) // 2 - int(7 * s); cx = x0 + (y1 - y0) // 2 + (x1 - x0 - (y1 - y0)) * ligado; cy = (y0 + y1) / 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
        txt = 'LIGADO' if ligado > 0.5 else 'DESLIGADO'
        w = d.textlength(txt, font=self.f_sobre)
        d.text(((x0 + x1) / 2 - w / 2, y1 + int(4 * s)), txt, font=self.f_sobre, fill=cor if ligado > 0.5 else (120, 112, 104))
        return im

    def colar(self, base, im, cx, cy, alfa=1.0, esc=1.0):
        if esc != 1.0:
            im = im.resize((max(1, int(im.width * esc)), max(1, int(im.height * esc))), Image.BICUBIC)
        if alfa < 1.0:
            a = im.getchannel('A').point(lambda v: int(v * alfa)); im = im.copy(); im.putalpha(a)
        base.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

    def entra_sai(self, t, ini, fim, dur_in=0.35, dur_out=0.3):
        if t < ini or t > fim: return 0.0, 0.0
        return ease((t - ini) / dur_in) * (1 - ease((t - (fim - dur_out)) / dur_out)), ease((t - ini) / dur_in)

    # --- legenda karaokê ---
    def legenda(self, base, t):
        if 14.95 <= t <= 16.6: return   # o original mostra "NUTRIGENÔMICA" escrito nesse trecho
        bl = next((b for b in self.leg if b['ini'] <= t < b['fim']), None)
        if not bl: return
        s = self.s; d = ImageDraw.Draw(base)
        ativa = max([i for i, p in enumerate(bl['pal']) if p['a'] - 0.04 <= t] or [0])
        ws = [p['w'] for p in bl['pal']]; esp = d.textlength(' ', font=self.f_leg)
        larg = [d.textlength(w, font=self.f_leg) for w in ws]
        total = sum(larg) + esp * (len(ws) - 1)
        x = (self.W - total) / 2; y = self.H * 0.735
        entrada = ease((t - bl['ini']) / 0.12)
        y += (1 - entrada) * 14 * s
        for i, w in enumerate(ws):
            if i == ativa:
                px, py = 16 * s, 8 * s
                bb = d.textbbox((x, y), w, font=self.f_leg, anchor='ls')
                d.rounded_rectangle([bb[0] - px, bb[1] - py, bb[2] + px, bb[3] + py + 4 * s], int(16 * s), fill=VERDE + (255,))
            d.text((x, y), w, font=self.f_leg, fill=(255, 255, 255, 255), anchor='ls',
                   stroke_width=int(4 * s) if i != ativa else 0, stroke_fill=(30, 22, 18, 150))
            x += larg[i] + esp

    # --- quadro completo ---
    def desenhar(self, base, t):
        s, W, H = self.s, self.W, self.H
        a, k = self.entra_sai(t, *self.ev['gancho'])
        if a > 0:
            self.colar(base, self.cartao('VOCÊ SABIA?', 'Seu prato conversa\ncom seus genes.'), W / 2, H * 0.17 - (1 - k) * 40 * s, a)
        a, k = self.entra_sai(t, *self.ev['epi'])
        if a > 0:
            self.colar(base, self.cartao('EPIGENÉTICA', 'Não muda o DNA.', 'Muda quais genes ficam ligados.'), W / 2, H * 0.15, a, 0.9 + 0.1 * ease_back(k))
        ini, fim, t_on, t_off = self.ev['interruptor']
        a, k = self.entra_sai(t, ini, fim)
        if a > 0:
            lig = ease((t - t_on) / 0.18) * (1 - ease((t - t_off) / 0.18))
            self.colar(base, self.interruptor(lig, VERDE_ON, 'GENE'), W / 2, H * 0.14, a, 0.9 + 0.1 * ease_back(k))
        ini, fim, ts = self.ev['ruins']
        if ini <= t <= fim:
            sai = 1 - ease((t - (fim - 0.3)) / 0.3)
            for (txt, ang, yy), tt in zip([('Ultraprocessados', -4, 0.12), ('Açúcares', 3, 0.20), ('Gorduras', -2, 0.28)], ts):
                if t >= tt:
                    kk = (t - tt) / 0.32
                    self.colar(base, self.etiqueta(txt, VERMELHO, ang), W / 2, H * yy, ease(kk) * sai, 0.6 + 0.4 * ease_back(kk))
        ini, fim, t_on = self.ev['gene_ruim']
        a, k = self.entra_sai(t, ini, fim)
        if a > 0:
            self.colar(base, self.interruptor(ease((t - t_on) / 0.18), VERMELHO, 'GENE'), W / 2, H * 0.14, a, 0.9 + 0.1 * ease_back(k))
        a, k = self.entra_sai(t, *self.ev['cta'], dur_out=0.01)
        if a > 0:
            self.colar(base, self.cartao('SALVE ESTE VÍDEO', 'Sua saúde começa\nno prato.', 'Mande para quem precisa ouvir isso.'), W / 2, H * 0.17 - (1 - k) * 40 * s, a)
        self.legenda(base, t)


# ---------------- vídeo ----------------
def ler_quadros(original):
    cor = 'eq=contrast=1.05:saturation=1.07:gamma=0.98,colorbalance=rm=0.015:bm=-0.015'
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', original, '-vf', cor, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         stdout=subprocess.PIPE)
    n = SW * SH * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(SH, SW, 3)


class Ampliador:
    def __init__(self):
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = os.cpu_count() or 4
        self.s = ort.InferenceSession(os.path.join(MOD, 'upscale', 'realesr-general-x4v3.onnx'), o, providers=['CPUExecutionProvider'])

    def __call__(self, im):
        x = np.ascontiguousarray((np.asarray(im, np.float32) / 255).transpose(2, 0, 1)[None])
        y = self.s.run(None, {'entrada': x})[0][0]
        return Image.fromarray((np.clip(y.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8))


def quadro(src, t, cortes, graf, W, ampliar):
    H = graf.H
    x0, y0, w, h = recorte_no_tempo(cortes, t)
    im = Image.fromarray(src).crop((x0 * SW, y0 * SH, (x0 + w) * SW, (y0 + h) * SH))
    if ampliar:
        im = ampliar(im.resize((W // 4, H // 4), Image.LANCZOS))
    else:
        im = im.resize((W, H), Image.LANCZOS)
    base = im.convert('RGBA')
    graf.desenhar(base, t)
    return base.convert('RGB')


# ---------------- áudio ----------------
def efeitos_sonoros(graf, dur, sr=48000):
    rng = np.random.default_rng(3); out = np.zeros(int(dur * sr), np.float32)

    def whoosh(t0, d=0.45, vol=0.05):
        n = int(d * sr); r = rng.standard_normal(n).astype(np.float32)
        f = np.fft.rfft(r); fr = np.fft.rfftfreq(n, 1 / sr); f *= np.exp(-((fr - 2500) / 2200) ** 2); r = np.fft.irfft(f, n)
        env = np.sin(np.linspace(0, np.pi, n)) ** 2; r = r / np.abs(r).max() * env * vol
        i = int(t0 * sr); out[i:i + n] += r[:len(out) - i]

    def pop(t0, vol=0.09):
        n = int(0.09 * sr); tt = np.arange(n) / sr
        fq = 950 * np.exp(-tt * 9); s_ = np.sin(2 * np.pi * np.cumsum(fq) / sr) * np.exp(-tt * 45) * vol
        i = int(t0 * sr); out[i:i + n] += s_[:len(out) - i]

    def clique(t0, vol=0.07):
        n = int(0.03 * sr); tt = np.arange(n) / sr
        s_ = rng.standard_normal(n) * np.exp(-tt * 300) * vol
        i = int(t0 * sr); out[i:i + n] += s_[:len(out) - i]

    e = graf.ev
    whoosh(e['gancho'][0] - 0.05); pop(e['epi'][0]); pop(e['interruptor'][0])
    clique(e['interruptor'][2]); clique(e['interruptor'][3])
    for tt in e['ruins'][2]: pop(tt)
    pop(e['gene_ruim'][0]); clique(e['gene_ruim'][2]); whoosh(e['cta'][0] - 0.05)
    return out


def mixar(voz_wav, fundo_wav, graf, dur, saida_wav):
    from pedalboard import Pedalboard, HighpassFilter, PeakFilter, Compressor
    sr = 48000
    v = carregar(voz_wav, sr, 2); f = carregar(fundo_wav, sr, 2)
    n = min(v.shape[1], f.shape[1], int(dur * sr)); v, f = v[:, :n], f[:, :n]
    v = Pedalboard([HighpassFilter(80), PeakFilter(250, -2.0, 1.0), PeakFilter(3800, 2.0, 0.9),
                    Compressor(threshold_db=-22, ratio=2.5, attack_ms=8, release_ms=90)])(v.astype(np.float32), sr)
    sfx = efeitos_sonoros(graf, n / sr, sr)[:n]
    mix = v * 1.12 + f * 0.75 + sfx[np.newaxis, :]
    sys.path.insert(0, os.path.join(AQUI, '..'))
    import tratar_audio as ta
    ta.SR = sr
    mix = ta.finalizar(mix.astype(np.float32), -14, -1.5)
    salvar_wav(saida_wav, mix, sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original'); ap.add_argument('voz'); ap.add_argument('fundo')
    ap.add_argument('--modo', choices=['previa', 'final'], default='previa'); ap.add_argument('--saida', required=True)
    ap.add_argument('--quadros', help='tempos separados por vírgula: só exporta esses quadros em PNG')
    a = ap.parse_args()
    W = 2160 if a.modo == 'final' else 1080
    P = tempos_palavras(a.voz); cortes = plano_de_cortes(P); graf = Grafismos(W, P)
    ampliar = Ampliador() if a.modo == 'final' else None
    if a.quadros:
        os.makedirs(a.saida, exist_ok=True)
        alvos = sorted(float(x) for x in a.quadros.split(','))
        for i, src in enumerate(ler_quadros(a.original)):
            t = i / FPS
            if alvos and t >= alvos[0]:
                quadro(src, t, cortes, graf, W, ampliar).save(os.path.join(a.saida, f'q_{t:05.1f}.png')); alvos.pop(0)
            if not alvos: break
        return
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', a.original],
                               capture_output=True, text=True).stdout)
    tmp = tempfile.mkdtemp()
    print('áudio…', flush=True)
    mixar(a.voz, a.fundo, graf, dur, f'{tmp}/audio.wav')
    print('vídeo…', flush=True)
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{graf.H}',
                            '-r', str(FPS), '-i', '-', '-i', f'{tmp}/audio.wav', '-map', '0:v', '-map', '1:a',
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '17' if a.modo == 'final' else '20',
                            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-c:a', 'aac', '-b:a', '256k', '-shortest',
                            '-movflags', '+faststart', a.saida], stdin=subprocess.PIPE)
    total = int(dur * FPS)
    for i, src in enumerate(ler_quadros(a.original)):
        enc.stdin.write(quadro(src, i / FPS, cortes, graf, W, ampliar).tobytes())
        if i % 60 == 0: print(f'quadro {i}/{total}', flush=True)
    enc.stdin.close(); enc.wait()
    print('pronto', a.saida)


if __name__ == '__main__':
    main()
