#!/usr/bin/env bash
# Instala o kit de áudio: bibliotecas Python e modelos (~3,5 GB). Pode rodar de novo sem problema; só baixa o que falta.
set -e
cd "$(dirname "$0")"
mkdir -p modelos
pip install -q -r requisitos.txt
G=https://github.com/k2-fsa/sherpa-onnx/releases/download
baixa() { [ -e "modelos/$2" ] && { echo "ok  $2"; return; }; echo "baixando $2…"; (cd modelos && curl -fsSL "$G/$1" | tar xj); }
baixa_arq() { [ -e "modelos/$2" ] && { echo "ok  $2"; return; }; echo "baixando $2…"; curl -fsSL -o "modelos/$2" "$G/$1"; }
baixa source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2 sherpa-onnx-spleeter-2stems-fp16 &
baixa audio-tagging-models/sherpa-onnx-ced-base-audio-tagging-2024-04-19.tar.bz2 sherpa-onnx-ced-base-audio-tagging-2024-04-19 &
baixa asr-models/sherpa-onnx-whisper-medium.tar.bz2 sherpa-onnx-whisper-medium &
baixa_arq speech-enhancement-models/gtcrn_simple.onnx gtcrn_simple.onnx &
baixa_arq asr-models/silero_vad.onnx silero_vad.onnx &
mkdir -p modelos/kokoro
K=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
[ -e modelos/kokoro/kokoro-v1.0.onnx ] || curl -fsSL -o modelos/kokoro/kokoro-v1.0.onnx $K/kokoro-v1.0.onnx &
[ -e modelos/kokoro/voices-v1.0.bin ] || curl -fsSL -o modelos/kokoro/voices-v1.0.bin $K/voices-v1.0.bin &
mkdir -p modelos/upscale modelos/fontes
[ -e modelos/upscale/realesr-general-x4v3.onnx ] || { curl -fsSL -o modelos/upscale/realesr-general-x4v3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth && python3 ferramentas-video/esrgan_onnx.py modelos/upscale/realesr-general-x4v3.pth modelos/upscale/realesr-general-x4v3.onnx; } &
if [ ! -d modelos/fontes/package ]; then (cd modelos/fontes && for p in montserrat dm-serif-display; do t=$(curl -fsSL https://registry.npmjs.org/@fontsource/$p/latest | python3 -c "import sys,json;print(json.load(sys.stdin)['dist']['tarball'])"); curl -fsSL "$t" | tar xz --wildcards "package/files/*latin-[0-9]*-normal.woff"; done) & fi
wait
command -v ffmpeg >/dev/null || echo "ATENÇÃO: instale o ffmpeg (apt install ffmpeg / brew install ffmpeg)"
echo "Kit pronto."
