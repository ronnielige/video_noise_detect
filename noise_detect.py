import cv2
import numpy as np

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

            # 若纹理太强，跳过
            #print(grad_mean)
            if grad_mean > texture_thresh:
                continue

            # 块均值差（代表整体位移）
            mean1 = np.mean(block1)
            mean2 = np.mean(block2)
            sad_block = abs(mean1 - mean2) * BLOCK_SIZE * BLOCK_SIZE

            # 每像素差值（排除微弱跳动）
            pixel_diff = np.abs(block1.astype(np.int16) - block2.astype(np.int16))
            pixel_diff[pixel_diff <= 2] = 0
            sum_diff = np.sum(pixel_diff)

            sad_total += sad_block
            sum_total += sum_diff
            valid_block_count += 1

    # 若没有有效块，返回 0
    if valid_block_count == 0:
        return 0.0

    # 防除以零 + 避免负值
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

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        noise_score = compute_temporal_noise(prev_gray, curr_gray)

        # 显示 + 保存
        overlay_text(curr_frame,
                     f"Temporal Noise: {noise_score:.2f}",
                     pos=(width - 350, 40))

        out.write(curr_frame)
        print(f"[Frame {frame_idx:04d}] Noise = {noise_score:.2f}")

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    out.release()
    print(f"输出已保存为：{output_path}")

if __name__ == "__main__":
    process_video_temporal_noise("cam0rec_ep39_jian_1.mp4", "cam0rec_ep39_jian_1_noise_overlay2.ts")
