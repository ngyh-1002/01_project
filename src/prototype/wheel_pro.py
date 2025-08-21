# ============================
# Roboflow Wheelchair Detection (class_id=0,1,60,92만 탐지)
# ============================

from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt

# 1. 클라이언트 초기화
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3ozt88YJNcSymJj72sF5"
)

# 2. 이미지 경로 지정
image_path = "../../assets/wheelchair/test/images/detect/2d67hyy_jpg_4bb8703a90a5f8f9a5529ae9829bee4a.jpg"

# 3. 추론 실행
result = CLIENT.infer(image_path, model_id="wheelchair-detection-hh3io-gpvvj/1")
print("추론 결과:", result)

# 4. 결과 시각화
image = cv2.imread(image_path)

# 관심있는 class_id
target_classes = [0, 1, 60, 92]

for pred in result['predictions']:
    if pred.get('class_id') in target_classes:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        confidence = pred['confidence']

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)

        # 텍스트 내용: class_id와 confidence
        text = f"class:{pred['class_id']} ({confidence:.2f}) x:{x} y:{y} w:{w} h:{h}"
        cv2.putText(image, text, (x - w//2, y - h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# BGR → RGB 변환 후 matplotlib으로 표시
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
