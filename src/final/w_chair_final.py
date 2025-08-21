import cv2
from ultralytics import YOLO
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# =========================
# 1. YOLO 모델 로드
# =========================
model_path = "best.pt"
model = YOLO(model_path)

# =========================
# 2. 동영상 경로 설정
# =========================
video_path = "../../assets/wheelchair/test/images/2025_08_21_16_30_mosaic_yolo.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"동영상 파일을 열 수 없습니다: {video_path}")

# =========================
# 3. 결과 저장/출력 옵션
# =========================
output_dir = "./output_frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
detected_count = 0

# =========================
# 4. 휠체어 대기 ROI 설정
# =========================
roi_x1, roi_y1 = 450, 450
roi_x2, roi_y2 = 750, 650

slope_fill_y_start = 719
slope_fill_y_end = roi_y2
slope_fill_progress = slope_fill_y_start

# 한글 폰트 경로 (Windows)
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 32)

# 슬로프 애니메이션 속도 계수
slope_speed = 0.05  # 0~1, 클수록 빨라짐

# =========================
# 5. 프레임별 탐지
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.predict(frame, imgsz=640, conf=0.25)

    detected = False
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, box)

                # ROI와 겹치는 면적 계산
                overlap_x = max(0, min(x2, roi_x2) - max(x1, roi_x1))
                overlap_y = max(0, min(y2, roi_y2) - max(y1, roi_y1))
                overlap_area = overlap_x * overlap_y
                roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)

                if roi_area > 0 and (overlap_area / roi_area) > 0.3:  # 30% 이상 겹침
                    detected = True

                # YOLO 박스 그리기
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ROI 박스 그리기
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    # PIL을 이용해 한글 텍스트 추가
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    if detected:
        detected_count += 1
        draw.text((50, 50), "휠체어 고객 감지! 슬로프를 작동합니다!",
                  font=font, fill=(255, 0, 0))

        # 애니메이션 박스: 현재 위치에서 목표 위치까지 선형 보간
        slope_fill_progress = slope_fill_progress + (slope_fill_y_end - slope_fill_progress) * slope_speed
        draw.rectangle([roi_x1, slope_fill_progress, roi_x2, slope_fill_y_start], fill=(255, 0, 0))

    else:
        # 감지 안되면 다시 초기 위치로 천천히 복원
        slope_fill_progress = slope_fill_progress + (slope_fill_y_start - slope_fill_progress) * slope_speed

    # OpenCV로 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Wheelchair Detection & Slope", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if detected:
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.jpg"), frame)

cap.release()
cv2.destroyAllWindows()
print(f"총 프레임 수: {frame_count}")
print(f"휠체어가 탐지된 프레임 수: {detected_count}")
