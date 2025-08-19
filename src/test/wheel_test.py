from inference_sdk import InferenceHTTPClient
import cv2

# ===========================
# 1. Roboflow API 클라이언트 설정
# ===========================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3ozt88YJNcSymJj72sF5"
)

MODEL_ID = "wheelchair-detection-hh3io/3"

# ===========================
# 2. 이미지 휠체어 감지 및 ROI 표시
# ===========================
def detect_wheelchair(image_path):
    # Roboflow API 호출 (경로 전달)
    result = CLIENT.infer(image_path, model_id=MODEL_ID)
    predictions = result["predictions"]

    # OpenCV로 이미지 읽기
    img = cv2.imread(image_path)

    # ROI 박스 그리기
    for pred in predictions:
        if pred["class"] == "wheelchair":
            x1 = int(pred["x"])
            y1 = int(pred["y"])
            x2 = int(pred["x"] + pred["width"])
            y2 = int(pred["y"] + pred["height"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Wheelchair {pred['confidence']:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # 결과 출력
    cv2.imshow("Wheelchair Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===========================
# 3. 실행
# ===========================
if __name__ == "__main__":
    detect_wheelchair("../../assets/wheelchair/test/images/2j8mpth_jpg_926f63507d4b01a491e8122679c5c0ec.jpg")  # 여기에 검사할 이미지 경로