# Contexto para o Claude

Este repositório guarda o que o Claude aprendeu e construiu com o dono da conta (Cortes Company Mídia), que cria vídeos curtos para redes sociais. Leia isto antes de começar.

## Como trabalhar com este usuário

- Fale sempre em **português do Brasil**, de forma simples e direta. O usuário não é programador: explique passos de interface (onde clicar) em vez de termos técnicos.
- Ele pede **eficiência**: instale o que precisar sem perguntar, rode tarefas longas em segundo plano e dê notícias curtas de progresso.
- Ele quer **aprender e ensinar**, não copiar: ao analisar referências, extraia princípios e não reproduza o conteúdo dos outros criadores.
- Ao publicar páginas, não use imagens de pessoas reais tiradas das referências; desenhe maquetes.

## O que já existe aqui

| Caminho | O que é |
|---|---|
| `manuais/aprendizados.md` | **Leia primeiro.** Resumo em texto de todas as regras de edição aprendidas |
| `manuais/manual-edicao-vertical.html` | Manual de vídeos verticais do dia a dia (publicado em https://claude.ai/artifact/Kof22isVNLBQzg2zL3qDTD) |
| `manuais/manual-showreel.html` | Capítulo de showreels, para momentos especiais (https://claude.ai/artifact/C7kgwu2ShstwUC8M7FTYaQ) |
| `mapa_sonoro.py` | "Escuta" um vídeo: separa voz e fundo, transcreve, nomeia efeitos, lê a música e cruza com os cortes |
| `tratar_audio.py` | Pós-produção de voz: limpeza, equalização, compressão, ducking e loudness −14 LUFS |
| `ferramentas-video/analisar_video.py` | Ritmo de cortes, gancho, folhas de quadros e medidas de áudio de um vídeo de referência |
| `ferramentas-video/narrar.py` | Narração em pt-BR com voz sintética (Kokoro: pm_alex, pm_santa, pf_dora) |
| `mapas-sonoros/` | Mapas sonoros dos 8 vídeos de referência (5 verticais, 3 showreels). Gerados antes da correção de nomes de efeitos e do BPM; refaça com `mapa_sonoro.py` se precisar de precisão |
| `exemplos/editar_nutrigenomica.py` | **Modelo de reedição completa** de um vídeo vertical: zoom que esconde legenda antiga, cortes no ritmo das frases, legenda karaokê sincronizada, cartões, som tratado e 4K com IA. Use como base para os próximos |
| `ferramentas-video/esrgan_onnx.py` | Converte o Real-ESRGAN para ONNX (aumento de resolução 4× com IA, sem PyTorch) |
| `exemplos/` (outros) | O primeiro vídeo feito (animação de texto com PIL + ffmpeg) e a narração dele |

## Preparar o ambiente

```bash
./instalar.sh   # bibliotecas + ~3,5 GB de modelos em modelos/ (ignorado pelo git)
```

Lições sobre o ambiente na nuvem:
- **GitHub e PyPI funcionam.** Hugging Face, YouTube, servidores do Whisper/OpenAI, Vosk e Google TTS são bloqueados.
- Por isso todos os modelos vêm de releases do GitHub: `k2-fsa/sherpa-onnx` (Whisper, Spleeter, CED, GTCRN, Silero VAD) e `thewh1teagle/kokoro-onnx` (voz).
- Vídeos do YouTube não abrem: peça ao usuário o arquivo mp4.
- Não há `apt-get`. ffmpeg já vem instalado.
- PyTorch não instala (o servidor dele é bloqueado): por isso o Real-ESRGAN roda em ONNX.
- Arquivos soltos do GitHub (raw) são bloqueados; fontes vêm do npm (`@fontsource/montserrat`, `@fontsource/dm-serif-display`).
- 4K com IA leva ~4 s por quadro (cerca de 2 h para 1 minuto a 30 qps). Faça antes uma prévia em 1080p (~1,5 min) e só depois o 4K. Se a sessão ficar inativa, a renderização para.

## Limites (seja honesto com o usuário sobre eles)

- O Claude **não ouve** áudio: tudo sobre som vem de modelos e medidas. Peça ao usuário para ouvir o resultado.
- O classificador de sons nomeia mal efeitos sintéticos ("sonar" costuma ser ping eletrônico; "heart murmur" costuma ser sub grave). Num corte, "veículo" costuma ser whoosh.
- O BPM não é confiável quando a música está muito baixa sob a fala (o detector tende a dar 120).
- Não gera imagens realistas nem vídeo com pessoas. O forte é texto, gráfico, animação e montagem de material enviado.

## Trabalhos feitos

- Vídeo de eleição 2026 (teste de animação e narração).
- Manuais de edição vertical e de showreel, a partir de 8 vídeos de referência.
- Reedição do vídeo de nutrigenômica de uma amiga do usuário: prévia em 1080p entregue ao usuário; o 4K foi interrompido e não terminou.

## Próximos passos combinados

- Testar o `tratar_audio.py` numa gravação real do usuário (ainda não feito; o teste foi com voz sintética degradada).
- Usar os manuais para produzir vídeos no estilo aprendido.
