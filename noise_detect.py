import cv2
import numpy as np
import os
import time
from glob import glob

BLOCK_SIZE = 16

def compute_temporal_noise(prev_gray, curr_gray, texture_thresh=50):
    h, w = curr_gray.shape
    sad_total = 0
    sum_total = 0
    valid_block_count = 0

    for y in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            block1 = prev_gray[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            block2 = curr_gray[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

            # 计算块纹理强度（可用方差或梯度）
            sobelx = cv2.Sobel(block2, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(block2, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            grad_mean = np.mean(grad_mag)

            if grad_mean > texture_thresh:
                continue

            mean1 = np.mean(block1)
            mean2 = np.mean(block2)
            sad_block = abs(mean1 - mean2) * BLOCK_SIZE * BLOCK_SIZE

            pixel_diff = np.abs(block1.astype(np.int16) - block2.astype(np.int16))
            pixel_diff[pixel_diff <= 2] = 0
            sum_diff = np.sum(pixel_diff)

            sad_total += sad_block
            sum_total += sum_diff
            valid_block_count += 1

    if valid_block_count == 0:
        return 0.0

    eps = 1e-3
    noise_score = max((sum_total - sad_total), 0) / max(sad_total, eps)
    return noise_score


def overlay_text(frame, text, pos=(10, 30), color=(0, 0, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)

def process_video_temporal_noise(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取第一帧")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 1
    total_time = 0.0
    max_frames = 300

    while frame_idx < max_frames:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        start_time = time.time()
        noise_score = compute_temporal_noise(prev_gray, curr_gray)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        total_time += elapsed_ms

        overlay_text(curr_frame,
                     f"Temporal Noise: {noise_score:.2f}",
                     pos=(width - 350, 40))
        out.write(curr_frame)

        print(f"[Frame {frame_idx:04d}] Noise = {noise_score:.2f}, Time = {elapsed_ms:.2f} ms")

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    out.release()

    avg_time = total_time / (frame_idx - 1) if frame_idx > 1 else 0
    print(f"\n🎬 处理完成：{os.path.basename(input_path)}")
    print(f"📏 分辨率: {width}x{height}")
    print(f"🧮 分析帧数: {frame_idx - 1}")
    print(f"⏱️ 每帧噪点检测平均耗时: {avg_time:.2f} ms")
    print(f"📁 输出已保存为：{output_path}\n")

def batch_process(input_dir="videos", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    exts = [".mp4", ".ts", ".avi", ".mkv", ".mov"]
    files = [f for f in glob(os.path.join(input_dir, "*")) if os.path.splitext(f)[-1].lower() in exts]

    if not files:
        print("⚠️ 未找到任何视频文件")
        return

    for file in files:
        base = os.path.basename(file)
        name, ext = os.path.splitext(base)
        out_file = os.path.join(output_dir, f"{name}_noise{ext}")
        process_video_temporal_noise(file, out_file)

if __name__ == "__main__":
    batch_process("videos", "output")
