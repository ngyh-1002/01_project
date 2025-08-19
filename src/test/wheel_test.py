from inference_sdk import InferenceHTTPClient
import cv2
import os
import glob

# ===========================
# 1. Roboflow API 클라이언트 설정
# ===========================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3ozt88YJNcSymJj72sF5"
)

MODEL_ID = "wheelchair-detection-hh3io/3"

# ===========================
# 2. 파일 이름 변경 및 이미지 검사
# ===========================
def detect_wheelchair_in_folder(folder_path):
    # 모든 이미지 파일 검색 (jpg, png, rf 포함)
    image_files = glob.glob(os.path.join(folder_path, "*.*"))

    no_wheelchair_images = []  # 휠체어 미검출 이미지 리스트

    for image_path in image_files:
        # 파일 이름에 '.rf.'가 있으면 제거 후 .jpg로 변경
        base_dir = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        if ".rf." in filename:
            new_filename = filename.replace(".rf.", "_")  # rf 제거
            if not new_filename.lower().endswith(".jpg"):
                new_filename += ".jpg"
            new_path = os.path.join(base_dir, new_filename)
            os.rename(image_path, new_path)
            image_path = new_path

        # OpenCV로 이미지 열기 확인
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] 이미지를 열 수 없습니다: {image_path}")
            continue

        # Roboflow API 호출
        try:
            result = CLIENT.infer(image_path, model_id=MODEL_ID)
            predictions = result["predictions"]

            wheelchair_found = any(pred["class"] == "wheelchair" for pred in predictions)

            if not wheelchair_found:
                print(f"[NO WHEELCHAIR] {image_path}")
                no_wheelchair_images.append(image_path)

                # 이미지 표시
                cv2.imshow("No Wheelchair Detected", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"[ERROR] {image_path} -> {e}")

    print(f"\n총 {len(no_wheelchair_images)}개의 이미지에서 휠체어 미검출")
    return no_wheelchair_images

# ===========================
# 3. 실행
# ===========================
if __name__ == "__main__":
    folder_path = "../../assets/wheelchair/test/images"
    detect_wheelchair_in_folder(folder_path)