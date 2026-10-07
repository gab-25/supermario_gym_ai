# supermario_gym_ai

Super Mario Bros (NES) giocato **zero-shot**, senza addestramento, da [NVIDIA NitroGen](https://huggingface.co/nvidia/NitroGen): un modello vision-to-action che legge il frame di gioco e restituisce i comandi del gamepad.

## Come funziona

1. Il frame NES (256×240) viene ridimensionato a 256×256 e passato a NitroGen, che gira in CPU.
2. NitroGen restituisce un blocco di 18 azioni gamepad, una per frame a 60 fps.
3. Ogni azione viene tradotta in un byte del controller NES:
   - stick o croce direzionale → destra/sinistra/giù
   - SOUTH/EAST → A (salto)
   - WEST/NORTH → B (corsa)
4. Le azioni vengono eseguite in `gym-super-mario-bros`, poi si ripete dal punto 1.

Il checkpoint (`ng.pt`, ~2 GB) viene scaricato da Hugging Face al primo avvio.

## Uso

```bash
uv sync
uv run supermario-gym-ai                          # 1 partita su 1-1, con finestra
uv run supermario-gym-ai --episodes 5 --no-render
uv run supermario-gym-ai --execute 6              # ripianifica ogni 6 frame invece di 18
uv run supermario-gym-ai --video partita.mp4
```

Ogni chiamata al modello richiede circa 3.5 s su una CPU a 8 core e copre 0.3 s di gioco.

## Licenza del modello

NitroGen è distribuito con la [NVIDIA Non-Commercial License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf): solo uso non commerciale.
