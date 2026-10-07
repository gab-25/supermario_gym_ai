import argparse
import time
from concurrent.futures import ThreadPoolExecutor

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


# --- NES BUTTONS (nes-py controller byte) ---
NES_A = 0b00000001
NES_B = 0b00000010
NES_DOWN = 0b00100000
NES_LEFT = 0b01000000
NES_RIGHT = 0b10000000

BUTTON_PRESS_THRES = 0.5
STICK_THRES = 0.5
BTN = {name: i for i, name in enumerate(BUTTON_ACTION_TOKENS)}


# --- MODEL (CPU version of nitrogen.inference_session) ---
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

    def predict(self, frame_rgb, num_samples=1):
        """RGB frame (H, W, 3) -> list of num_samples gamepad action chunks (j_left, j_right, buttons)."""
        image = Image.fromarray(cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_AREA))
        pixel_values = self.img_proc([image], return_tensors="pt")["pixel_values"].to(self.device)

        data = {
            "frames": pixel_values,
            "dropped_frames": torch.zeros((1,), dtype=torch.bool, device=self.device),
            "game": None,
        }
        # The same frame repeated along the batch: each sample starts from different noise
        tokenized = self.tokenizer.encode(data)
        for k, v in tokenized.items():
            if isinstance(v, np.ndarray):
                v = torch.tensor(v)
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.unsqueeze(0).expand(num_samples, *v.shape).to(self.device)
            else:
                tokenized[k] = [v] * num_samples

        with torch.inference_mode():
            pred = self.tokenizer.decode(self.model.get_action(tokenized))
        pred = {k: v.float().cpu().numpy() for k, v in pred.items()}
        return [{k: v[i] for k, v in pred.items()} for i in range(num_samples)]


# --- GAMEPAD -> NES ---
def to_nes_actions(pred):
    """Convert a chunk of gamepad actions into NES controller bytes."""
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
        # Jump: bottom face button (Xbox A) or right face button (Nintendo A)
        if pressed("SOUTH") or pressed("EAST"):
            byte |= NES_A
        # Run: left / top face buttons
        if pressed("WEST") or pressed("NORTH"):
            byte |= NES_B
        actions.append(byte)
    return actions


# --- EMULATOR LOOKAHEAD ---
def rollout_score(env, actions, tail_frames):
    """Play actions from the current state, then rewind the emulator.

    The chunk is followed by tail_frames with no buttons pressed, to catch
    deaths that are already decided but not yet visible (e.g. falling into a pit).
    Returns a sortable score: (flag, survives chunk, survives tail, x).
    """
    env._backup()
    info, died_in_chunk, died_in_tail = {}, False, False
    for byte in actions:
        _, _, done, info = env.step(byte)
        if done:
            died_in_chunk = not info.get("flag_get")
            break
    if not done:
        for _ in range(tail_frames):
            _, _, done, info = env.step(0)
            if done:
                died_in_tail = not info.get("flag_get")
                break
    env._restore()
    env.done = False
    return (bool(info.get("flag_get")), not died_in_chunk, not died_in_tail, info.get("x_pos", 0))


def describe(byte):
    names = [("RIGHT", NES_RIGHT), ("LEFT", NES_LEFT), ("DOWN", NES_DOWN), ("A", NES_A), ("B", NES_B)]
    return "+".join(n for n, b in names if byte & b) or "noop"


def run():
    parser = argparse.ArgumentParser(description="Super Mario Bros played zero-shot by NVIDIA NitroGen")
    parser.add_argument("--level", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--execute", type=int, default=18, help="Actions of each chunk executed before replanning (max 18)")
    parser.add_argument("--max-chunks", type=int, default=400)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--video", default=None, help="Save the gameplay to an .mp4 file")
    parser.add_argument("--samples", type=int, default=1,
                        help="Action chunks sampled per step; with >1 the emulator previews each and plays the best")
    parser.add_argument("--tail", type=int, default=30, help="No-input frames simulated after each previewed chunk")
    parser.add_argument("--stuck-steps", type=int, default=3,
                        help="Steps without progress after which the number of samples is doubled")
    parser.add_argument("--max-samples", type=int, default=16, help="Upper bound for samples when stuck")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    print("Loading NitroGen...")
    policy = NitroGenPolicy(hf_hub_download("nvidia/NitroGen", "ng.pt"))

    writer = None
    if args.video:
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 60, (256, 240))

    def timed_predict(frame, num_samples):
        t = time.time()
        pred = policy.predict(frame, num_samples=num_samples)
        return pred, time.time() - t

    # The model runs in a worker thread: while it computes the next chunk, the
    # window shows the frames of the chunk just emulated, spread over the
    # inference time (smooth slow motion instead of freeze + jump)
    executor = ThreadPoolExecutor(max_workers=1)

    for episode in range(args.episodes):
        # Raw NES env: action = controller byte, 1 step = 1 frame (60 fps)
        # .unwrapped strips gym's TimeLimit, incompatible with the old 4-value step API.
        # A fresh env per episode: the lookahead overwrites the emulator backup that
        # env.reset() would restore
        env = gym_super_mario_bros.make(args.level).unwrapped
        obs = env.reset()
        done, info, max_x, latencies = False, {}, 0, []
        num_samples, stalled = args.samples, 0
        pending = executor.submit(timed_predict, obs, num_samples)
        for chunk in range(args.max_chunks):
            preds, latency = pending.result()
            latencies.append(latency)
            candidates = [to_nes_actions(pred)[: args.execute] for pred in preds]
            if len(candidates) > 1:
                scores = [rollout_score(env, c, args.tail) for c in candidates]
                best = max(range(len(candidates)), key=lambda i: scores[i])
                safe = sum(s[1] and s[2] for s in scores)
            else:
                best, safe = 0, None
            actions = candidates[best]

            # The emulator runs the chunk instantly, frames are shown afterwards
            frames = []
            for byte in actions:
                obs, _, done, info = env.step(byte)
                frames.append(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                if done:
                    break
            if writer is not None:
                for frame_bgr in frames:
                    writer.write(frame_bgr)

            # When Mario stops advancing, ask NitroGen for more proposals so that
            # one of them is more likely to clear the obstacle
            x = info.get("x_pos", 0)
            if x > max_x:
                max_x, stalled = x, 0
            else:
                stalled += 1
            if args.samples > 1 and stalled >= args.stuck_steps:
                num_samples = min(args.samples * 2 ** (stalled - args.stuck_steps + 1), args.max_samples)
            else:
                num_samples = args.samples

            finished = done or info.get("flag_get") or chunk == args.max_chunks - 1
            if not finished:
                pending = executor.submit(timed_predict, obs, num_samples)

            if not args.no_render:
                delay_ms = max(1, int(latency * 1000 / len(frames)))
                for frame_bgr in frames:
                    cv2.imshow("Super Mario Bros - NitroGen", cv2.resize(frame_bgr, (768, 720), interpolation=cv2.INTER_NEAREST))
                    cv2.waitKey(delay_ms)

            print(f"[ep {episode} chunk {chunk:3d}] {latency:.2f}s  x={info.get('x_pos')}  "
                  f"lives={info.get('life')}  actions={describe(actions[0])}..{describe(actions[-1])}"
                  + (f"  safe={safe}/{len(candidates)} pick={best}" if safe is not None else ""))
            if finished:
                break

        print(f"Episode {episode}: max x={max_x}  flag={info.get('flag_get', False)}  "
              f"mean latency={np.mean(latencies):.2f}s")
        env.close()

    if writer is not None:
        writer.release()
    executor.shutdown()


if __name__ == "__main__":
    run()
