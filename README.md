# supermario_gym_ai

Super Mario Bros (NES) played **zero-shot**, with no training, by [NVIDIA NitroGen](https://huggingface.co/nvidia/NitroGen): a vision-to-action model that reads the game frame and outputs gamepad actions.

## How it works

1. The NES frame (256×240) is resized to 256×256 and fed to NitroGen, running on CPU.
2. NitroGen returns a chunk of 18 gamepad actions, one per frame at 60 fps.
3. Each action is mapped to an NES controller byte:
   - stick or D-pad → right/left/down
   - SOUTH/EAST → A (jump)
   - WEST/NORTH → B (run)
4. The actions are executed in `gym-super-mario-bros`, then the loop repeats from step 1.

With `--samples K` (K > 1) NitroGen proposes K different chunks for the same frame. The emulator saves its state, previews each chunk (plus a few no-input frames to catch deaths that are already decided, like falling into a pit), rewinds, and plays the best one: the flag first, then survival, then distance. The actions always come from NitroGen; the emulator only filters out the deadly ones. If Mario makes no progress for `--stuck-steps` steps (default 3), the number of proposals is doubled each step, up to `--max-samples` (default 16), so that one of them is more likely to clear the obstacle.

While the model computes the next chunk, the window plays the frames of the previous one in slow motion, so the display does not freeze.

The checkpoint (`ng.pt`, ~2 GB) is downloaded from Hugging Face on first run.

## Usage

```bash
uv sync
uv run supermario-gym-ai                          # 1 episode on 1-1, with window
uv run supermario-gym-ai --episodes 5 --no-render
uv run supermario-gym-ai --execute 6              # replan every 6 frames instead of 18
uv run supermario-gym-ai --video gameplay.mp4
uv run supermario-gym-ai --samples 4              # 4 proposals per step, previewed in the emulator
```

Each model call takes ~3.5–4.5 s on an 8-core CPU and covers 0.3 s of gameplay; with `--samples K` it takes roughly K times longer. The process uses ~2.6 GB of RAM.

## Model license

NitroGen is released under the [NVIDIA Non-Commercial License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf): non-commercial use only.
