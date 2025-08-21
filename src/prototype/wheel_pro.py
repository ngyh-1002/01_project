import cv2
from ultralytics import YOLO
import os

# =========================
# 1. YOLO 모델 로드
# =========================
model_path = "best.pt"  # 모델 경로
model = YOLO(model_path)

# =========================
# 2. 동영상 경로 설정
# =========================
video_path = "../../assets/wheelchair/test/images/2025_08_21_16_30_mosaic_yolo.mp4"

# OpenCV로 동영상 읽기
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
# 4. 프레임별 탐지
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # YOLO 모델로 탐지
    results = model.predict(frame, imgsz=640, conf=0.25)  # conf는 신뢰도 threshold

    # 탐지 결과 확인
    detected = False
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            detected = True
            # 박스 그리기
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if detected:
        detected_count += 1

    # 화면에 프레임 표시 (옵션, 필요 없으면 주석 처리)
    cv2.imshow("Wheelchair Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 탐지 프레임 저장
    if detected:
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.jpg"), frame)

cap.release()
cv2.destroyAllWindows()

# =========================
# 5. 결과 출력
# =========================
print(f"총 프레임 수: {frame_count}")
print(f"휠체어가 탐지된 프레임 수: {detected_count}")
