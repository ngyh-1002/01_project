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