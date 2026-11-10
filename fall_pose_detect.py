from ultralytics import YOLO
import cv2
import os
import csv

# -------------------------- 配置区 --------------------------
VIDEO_FOLDER = r"E:\x下载\f_mask_b_1\f_mask_b_1"
SAVE_VIDEO_FOLDER = r"E:\elderly_fall_project\result\video_out"
SAVE_CSV_PATH = r"E:\elderly_fall_project\result\pred_result.csv"

CONF_THRESHOLD = 0.5       # 人体检测置信度
KPT_CONF_THRESHOLD = 0.3   # 关键点最低置信，过滤枕头假人
FALL_CONFIRM_FRAMES = 3    # 连续多少帧满足条件才判定跌倒

# ✅ 修正！YOLO pose 17个关键点 0~16，删掉索引17
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (0, 15)
]
# -----------------------------------------------------------

os.makedirs(SAVE_VIDEO_FOLDER, exist_ok=True)
model = YOLO("yolo11n-pose.pt")

csv_file = open(SAVE_CSV_PATH, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["video_name", "frame_id", "raw_fall", "confirmed_fall", "conf"])

video_suffix = (".mp4", ".avi", ".mov", ".mkv")
video_list = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(video_suffix)]
print(f"一共找到 {len(video_list)} 个视频！")

for video_name in video_list:
    video_path = os.path.join(VIDEO_FOLDER, video_name)
    out_path = os.path.join(SAVE_VIDEO_FOLDER, video_name)
    print(f"===== 正在处理：{video_name} =====")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    fall_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        raw_fall = 0
        confirmed_fall = 0
        b_conf = 0.0  # 初始化，无检测时为0

        results = model(frame, conf=CONF_THRESHOLD)
        res = results[0]

        if res.keypoints is not None:
            boxes = res.boxes
            kpts_all = res.keypoints.xy.cpu().numpy()
            kpts_conf_all = res.keypoints.conf.cpu().numpy()

            for idx, box in enumerate(boxes):
                kpt_conf = kpts_conf_all[idx]
                # 过滤：关键点平均置信太低 → 枕头/被子误识别，直接跳过
                avg_kpt_conf = kpt_conf.mean()
                if avg_kpt_conf < KPT_CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                b_conf = box.conf[0].item()
                pk = kpts_all[idx]

                nose_y = pk[0][1]
                l_hip_y = pk[11][1]
                r_hip_y = pk[12][1]
                hip_y = (l_hip_y + r_hip_y)/2

                bw = x2 - x1
                bh = y2 - y1
                if bh <= 0:
                    continue
                wh_ratio = bw / bh
                head_hip_diff = abs(nose_y - hip_y) / bh

                # 跌倒判定规则：宽高比较大 + 头髋垂直距离小
                if wh_ratio > 0.75 and head_hip_diff < 0.42:
                    raw_fall = 1
                else:
                    raw_fall = 0

                # 连续帧确认跌倒
                if raw_fall == 1:
                    fall_count += 1
                else:
                    fall_count = 0
                if fall_count >= FALL_CONFIRM_FRAMES:
                    confirmed_fall = 1

                # 画框、骨架、标签
                if confirmed_fall:
                    label = f"FALL {b_conf:.2f}"
                    color = (0,0,255)
                else:
                    label = f"NORMAL {b_conf:.2f}"
                    color = (0,255,0)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)) , color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                for (p1,p2) in SKELETON:
                    if kpt_conf[p1]>KPT_CONF_THRESHOLD and kpt_conf[p2]>KPT_CONF_THRESHOLD:
                        cv2.line(frame, (int(pk[p1][0]),int(pk[p1][1])), (int(pk[p2][0]),int(pk[p2][1])), (255,0,255),2)
                for (kx,ky),c in zip(pk, kpt_conf):
                    if c>KPT_CONF_THRESHOLD:
                        cv2.circle(frame, (int(kx),int(ky)),4,(0,255,255),-1)

        csv_writer.writerow([video_name, frame_idx, raw_fall, confirmed_fall, b_conf])
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"{video_name} 处理完毕")

csv_file.close()
print("\n✅ 全部视频跑完！")
