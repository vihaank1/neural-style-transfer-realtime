import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
import torch
from transformer_net import TransformerNet

MODEL_RES = 128

def load_model(ckpt, device):
    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model

def frame_to_tensor(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (MODEL_RES, MODEL_RES), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)

def tensor_to_frame(t, out_size):
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, out_size, interpolation=cv2.INTER_LINEAR)

def run(video_path, ckpt, out_path, max_frames, batch_size, out_size, threads):
    device = torch.device("cpu")
    torch.set_num_threads(threads)
    model = load_model(ckpt, device)

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n = len(frames)
    print(f"loaded {n} frames from {video_path}")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, out_size)

    # ---- baseline: naive one-frame-at-a-time inference, grad enabled ----
    t0 = time.time()
    for f in frames[: min(40, n)]:
        t = frame_to_tensor(f, device)
        _ = model(t)  # grad tracking on (naive/default mode)
    baseline_elapsed = time.time() - t0
    baseline_fps = min(40, n) / baseline_elapsed

    # ---- optimized: inference_mode + batched forward passes ----
    t0 = time.time()
    outputs = []
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            batch_frames = frames[i:i + batch_size]
            tensors = torch.cat([frame_to_tensor(f, device) for f in batch_frames], dim=0)
            out = model(tensors)
            for j in range(out.shape[0]):
                outputs.append(tensor_to_frame(out[j:j+1], out_size))
    optimized_elapsed = time.time() - t0
    optimized_fps = n / optimized_elapsed

    for frame in outputs:
        writer.write(frame)
    writer.release()

    result = {
        "frames_processed": n,
        "threads": threads,
        "batch_size": batch_size,
        "baseline_fps_no_grad_off": round(baseline_fps, 2),
        "optimized_fps_inference_mode_batched": round(optimized_fps, 2),
        "speedup_x": round(optimized_fps / baseline_fps, 2),
        "output_video": out_path,
    }
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="/tmp/style_transfer/data/vtest.avi")
    p.add_argument("--ckpt", default="/tmp/style_transfer/checkpoints/style_mosaic.pt")
    p.add_argument("--out", default="/tmp/style_transfer/samples/stylized_vtest.mp4")
    p.add_argument("--max_frames", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--log", default="/tmp/style_transfer/checkpoints/pipeline_benchmark.json")
    args = p.parse_args()
    res = run(args.video, args.ckpt, args.out, args.max_frames, args.batch_size, (384, 288), args.threads)
    with open(args.log, "w") as f:
        json.dump(res, f, indent=2)
