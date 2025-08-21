import os
import glob
import cv2
from inference_sdk import InferenceHTTPClient
from concurrent.futures import ThreadPoolExecutor

# ===========================
# 1. Roboflow 타개발자 학습 API 설정
# ===========================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3ozt88YJNcSymJj72sF5"
)
MODEL_ID = "wheelchair-detection-hh3io/3"

# ===========================
# 2. 이미지 리사이즈 함수
# ===========================
def resize_image_for_inference(image_path, max_size=640):
    img = cv2.imread(image_path)
    if img is None:
        return None, image_path
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        temp_path = image_path + ".resized.jpg"
        cv2.imwrite(temp_path, resized_img)
        return resized_img, temp_path
    else:
        return img, image_path

# ===========================
# 3. 이미지 분석 (API 호출)
# ===========================
def analyze_image(image_path):
    img, inference_path = resize_image_for_inference(image_path)
    if img is None:
        print(f"[ERROR] 이미지 로딩 실패: {image_path}")
        return None
    try:
        result = CLIENT.infer(inference_path, model_id=MODEL_ID)
        predictions = result["predictions"]
        return any(pred["class"] == "wheelchair" for pred in predictions)
    except Exception as e:
        print(f"[ERROR] {image_path} -> {e}")
        return None

# ===========================
# 4. 이미지 전처리
# ===========================
def preprocess_images(folder_path):
    image_files = glob.glob(os.path.join(folder_path, "*.*"))
    total_images = len(image_files)
    processed_files = []

    for image_path in image_files:
        base_dir = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        if ".rf." in filename:
            new_filename = filename.replace(".rf.", "_")
            if not new_filename.lower().endswith(".jpg"):
                new_filename += ".jpg"
            new_path = os.path.join(base_dir, new_filename)
            os.rename(image_path, new_path)
            image_path = new_path
        processed_files.append(image_path)

    # OpenCV 로딩 가능한 이미지만 필터링
    valid_images = [f for f in processed_files if cv2.imread(f) is not None]
    return total_images, valid_images

# ===========================
# 5. detect 폴더 처리
# ===========================
def process_detect_folder(folder_path):
    total_images, valid_images = preprocess_images(folder_path)
    print(f"\n📁 [detect] 총 이미지 수: {total_images}")
    print(f"✅ 유효 이미지 수 (OpenCV 로딩 성공): {len(valid_images)}")

    no_wheelchair_images = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(analyze_image, valid_images)
        for img_path, result in zip(valid_images, results):
            if result is False:
                # print(f"[NO WHEELCHAIR] {img_path}")
                no_wheelchair_images.append(img_path)

    print(f"✅ 유효 이미지 {len(valid_images)}개 중에 {len(no_wheelchair_images)}개 휠체어검출")
    return no_wheelchair_images

# ===========================
# 6. not_detect 폴더 처리
# ===========================
def process_not_detect_folder(folder_path):
    total_images, valid_images = preprocess_images(folder_path)
    print(f"\n📁 [not_detect] 총 이미지 수: {total_images}")
    print(f"✅ 유효 이미지 수 (OpenCV 로딩 성공): {len(valid_images)}")

    wheelchair_detected_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(analyze_image, valid_images)
        for img_path, result in zip(valid_images, results):
            if result is True:
                # print(f"[WHEELCHAIR DETECTED] {img_path}")
                wheelchair_detected_count += 1

    print(f"⚠️ 휠체어 검출 이미지 수: {wheelchair_detected_count}")
    print(f"✅ 유효 이미지 {len(valid_images)}개 중에 {wheelchair_detected_count}개 휠체어검출")
    return wheelchair_detected_count

# ===========================
# 7. 메인 실행
# ===========================
if __name__ == "__main__":
    detect_folder = "../../assets/wheelchair/test/images/detect"
    not_detect_folder = "../../assets/wheelchair/test/images/not_detect"

    print("🔍 detect 폴더 처리 중...")
    detect_failures = process_detect_folder(detect_folder)

    print("\n🔍 not_detect 폴더 처리 중...")
    detected_count = process_not_detect_folder(not_detect_folder)
