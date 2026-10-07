import argparse
import time

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

import gym_super_mario_bros
from nitrogen.cfg import CkptConfig
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.mm_tokenizers import NitrogenTokenizer
from nitrogen.shared import BUTTON_ACTION_TOKENS


# --- BOTTONI NES (byte del controller di nes-py) ---
NES_A = 0b00000001
NES_B = 0b00000010
NES_DOWN = 0b00100000
NES_LEFT = 0b01000000
NES_RIGHT = 0b10000000

BUTTON_PRESS_THRES = 0.5
STICK_THRES = 0.5
BTN = {name: i for i, name in enumerate(BUTTON_ACTION_TOKENS)}


# --- MODELLO (versione CPU di nitrogen.inference_session) ---
class NitroGenPolicy:
    def __init__(self, ckpt_path, device="cpu"):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_config = CkptConfig.model_validate(checkpoint["ckpt_config"])
        ckpt_config.tokenizer_cfg.training = False

        self.device = torch.device(device)
        self.img_proc = AutoImageProcessor.from_pretrained(ckpt_config.model_cfg.vision_encoder_name)
        self.tokenizer = NitrogenTokenizer(ckpt_config.tokenizer_cfg)
        self.model = NitroGen(config=ckpt_config.model_cfg, game_mapping=self.tokenizer.game_mapping)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval().to(self.device)
        self.tokenizer.eval()

    def predict(self, frame_rgb):
        """Frame RGB (H, W, 3) -> chunk di azioni gamepad (j_left, j_right, buttons)."""
        image = Image.fromarray(cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_AREA))
        pixel_values = self.img_proc([image], return_tensors="pt")["pixel_values"].to(self.device)

        data = {
            "frames": pixel_values,
            "dropped_frames": torch.zeros((1,), dtype=torch.bool, device=self.device),
            "game": None,
        }
        tokenized = self.tokenizer.encode(data)
        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.unsqueeze(0).to(self.device)
            elif isinstance(v, np.ndarray):
                tokenized[k] = torch.tensor(v, device=self.device).unsqueeze(0)
            else:
                tokenized[k] = [v]

        with torch.inference_mode():
            pred = self.tokenizer.decode(self.model.get_action(tokenized))
        return {k: v.squeeze().float().cpu().numpy() for k, v in pred.items()}


# --- GAMEPAD -> NES ---
def to_nes_actions(pred):
    """Converte il chunk di azioni gamepad nei byte del controller NES."""
    actions = []
    for (lx, ly), buttons in zip(pred["j_left"], pred["buttons"]):
        pressed = lambda name: buttons[BTN[name]] > BUTTON_PRESS_THRES
        byte = 0
        if pressed("DPAD_RIGHT") or lx > STICK_THRES:
            byte |= NES_RIGHT
        elif pressed("DPAD_LEFT") or lx < -STICK_THRES:
            byte |= NES_LEFT
        if pressed("DPAD_DOWN"):
            byte |= NES_DOWN
        # Salto: tasto in basso (Xbox A) o a destra (layout Nintendo A)
        if pressed("SOUTH") or pressed("EAST"):
            byte |= NES_A
        # Corsa: tasti a sinistra / in alto
        if pressed("WEST") or pressed("NORTH"):
            byte |= NES_B
        actions.append(byte)
    return actions


def describe(byte):
    names = [("RIGHT", NES_RIGHT), ("LEFT", NES_LEFT), ("DOWN", NES_DOWN), ("A", NES_A), ("B", NES_B)]
    return "+".join(n for n, b in names if byte & b) or "noop"


def run():
    parser = argparse.ArgumentParser(description="Super Mario Bros giocato zero-shot da NVIDIA NitroGen")
    parser.add_argument("--level", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--execute", type=int, default=18, help="Azioni del chunk eseguite prima di ripianificare (max 18)")
    parser.add_argument("--max-chunks", type=int, default=400)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--video", default=None, help="Salva la partita in un file .mp4")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    print("Caricamento NitroGen...")
    policy = NitroGenPolicy(hf_hub_download("nvidia/NitroGen", "ng.pt"))

    # Env NES grezzo: azione = byte del controller, 1 step = 1 frame (60 fps)
    # .unwrapped toglie il TimeLimit di gym, incompatibile con la vecchia API a 4 valori
    env = gym_super_mario_bros.make(args.level).unwrapped
    writer = None
    if args.video:
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 60, (256, 240))

    for episode in range(args.episodes):
        obs = env.reset()
        done, info, max_x, latencies = False, {}, 0, []
        for chunk in range(args.max_chunks):
            t = time.time()
            actions = to_nes_actions(policy.predict(obs))[: args.execute]
            latencies.append(time.time() - t)

            for byte in actions:
                obs, _, done, info = env.step(byte)
                frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                if writer is not None:
                    writer.write(frame_bgr)
                if not args.no_render:
                    cv2.imshow("Super Mario Bros - NitroGen", cv2.resize(frame_bgr, (768, 720), interpolation=cv2.INTER_NEAREST))
                    cv2.waitKey(1)
                if done:
                    break

            max_x = max(max_x, info.get("x_pos", 0))
            print(f"[ep {episode} chunk {chunk:3d}] {latencies[-1]:.2f}s  x={info.get('x_pos')}  "
                  f"vite={info.get('life')}  azioni={describe(actions[0])}..{describe(actions[-1])}")
            if done or info.get("flag_get"):
                break

        print(f"Episodio {episode}: x massimo={max_x}  bandiera={info.get('flag_get', False)}  "
              f"latenza media={np.mean(latencies):.2f}s")

    if writer is not None:
        writer.release()
    env.close()


if __name__ == "__main__":
    run()
