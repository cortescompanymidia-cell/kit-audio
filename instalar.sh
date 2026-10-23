#!/usr/bin/env bash
# Instala o kit de áudio: bibliotecas Python e modelos (~3 GB). Pode rodar de novo sem problema; só baixa o que falta.
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
wait
command -v ffmpeg >/dev/null || echo "ATENÇÃO: instale o ffmpeg (apt install ffmpeg / brew install ffmpeg)"
echo "Kit pronto."
