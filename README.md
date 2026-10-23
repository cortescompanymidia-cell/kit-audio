# Kit de edição

Ferramentas e manuais para criar vídeos curtos com edição e som de nível profissional.

- **Manuais** (`manuais/`): o que faz um vídeo vertical segurar a atenção, e como montar um showreel para momentos especiais. Comece por `manuais/aprendizados.md`.
- **Som** (`mapa_sonoro.py`, `tratar_audio.py`): descrever tudo o que acontece no áudio de um vídeo e tratar a voz como numa pós-produção. Instruções em `LEIAME.md`.
- **Vídeo** (`ferramentas-video/`): analisar o ritmo de edição de uma referência e gerar narração em português.

## Instalar

```bash
./instalar.sh
```

Precisa de Python 3.10+ e ffmpeg. Os modelos (cerca de 3,5 GB) vão para `modelos/` e não ficam no repositório.

## Uso rápido

```bash
python3 ferramentas-video/analisar_video.py referencia.mp4   # ritmo, gancho, quadros
python3 mapa_sonoro.py referencia.mp4                        # fala, efeitos, música
python3 tratar_audio.py minha_gravacao.mp4 --musica trilha.mp3
python3 ferramentas-video/narrar.py "Texto da narração." --saida voz.wav
```
