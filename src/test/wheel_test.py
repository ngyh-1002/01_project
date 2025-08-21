import os
import glob
from ultralytics import YOLO
import cv2

# =========================
# 1. YOLO 모델 로드
# =========================
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)  # CPU 환경 기준

print("모델 클래스 이름:", model.names)

# =========================
# 2. 검사할 경로 설정
# =========================
folders = {
    "detect": "../../assets/wheelchair/test/images/detect",       # 휠체어 없는 이미지 확인
    "not_detect": "../../assets/wheelchair/test/images/not_detect" # 휠체어 감지된 이미지 확인
}

conf_levels = [0.1, 0.2, 0.3]  # 여러 confidence 단계

# =========================
# 3. 폴더별 검사
# =========================
results_summary = {}

for key, folder in folders.items():
    image_paths = glob.glob(os.path.join(folder, "*.*"))
    count = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        # CLAHE 대비 향상
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)
        img_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)

        detected = False
        for conf in conf_levels:
            results = model(img_rgb, conf=conf, imgsz=max(img.shape[:2]), device='cpu')
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.tolist()]
            if "wheelchair" in detected_classes:
                detected = True
                break

        # 폴더별 카운트 규칙
        if key == "detect" and not detected:
            count += 1  # 휠체어 없는 이미지 수
        elif key == "not_detect" and detected:
            count += 1  # 휠체어 감지된 이미지 수

    results_summary[key] = (count, len(image_paths))

# =========================
# 4. 결과 출력
# =========================
print(f"\n=== 검사 결과 ===")
print(f"detect 폴더: 휠체어 없는 이미지 {results_summary['detect'][0]} / {results_summary['detect'][1]}")
print(f"not_detect 폴더: 휠체어 감지된 이미지 {results_summary['not_detect'][0]} / {results_summary['not_detect'][1]}")
