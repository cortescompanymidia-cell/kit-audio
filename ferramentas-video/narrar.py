#!/usr/bin/env python3
"""Narração em português do Brasil com voz sintética (Kokoro, roda offline).

Uso:
  python3 narrar.py "Texto a ser narrado." --saida voz.wav
  python3 narrar.py roteiro.txt --voz pf_dora --velocidade 1.1
Vozes: pm_alex (masculina, padrão), pm_santa (masculina), pf_dora (feminina).
Dica: escreva números por extenso ("quarenta e sete") para a pronúncia sair certa.
"""
import argparse, os, sys
MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modelos', 'kokoro')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('texto', help='texto entre aspas ou caminho de um .txt')
    ap.add_argument('--voz', default='pm_alex', choices=['pm_alex', 'pm_santa', 'pf_dora'])
    ap.add_argument('--velocidade', type=float, default=1.05); ap.add_argument('--saida', default='narracao.wav')
    a = ap.parse_args()
    from kokoro_onnx import Kokoro
    import soundfile as sf
    texto = open(a.texto).read() if os.path.isfile(a.texto) else a.texto
    k = Kokoro(os.path.join(MOD, 'kokoro-v1.0.onnx'), os.path.join(MOD, 'voices-v1.0.bin'))
    som, sr = k.create(texto, voice=a.voz, speed=a.velocidade, lang='pt-br')
    sf.write(a.saida, som, sr)
    print(f'{a.saida}: {len(som) / sr:.1f} s')


if __name__ == '__main__':
    main()
