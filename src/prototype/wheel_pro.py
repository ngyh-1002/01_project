import cv2
import threading
import queue
from inference_sdk import InferenceHTTPClient

# =========================
# 1. Roboflow API 설정
# =========================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3ozt88YJNcSymJj72sF5"
)

MODEL_ID_1 = "wheelchair-detection-hh3io-gpvvj/1"
TARGET_CLASSES = [0, 1, 60, 55, 92]
MODEL_ID_2 = "wheelchair-detection-hh3io/3"

# =========================
# 2. 프레임 분석 함수
# =========================
def analyze_frame(frame):
    try:
        result1 = CLIENT.infer(frame, model_id=MODEL_ID_1)
        pred1 = [p for p in result1["predictions"] if p.get("class_id") in TARGET_CLASSES]

        result2 = CLIENT.infer(frame, model_id=MODEL_ID_2)
        pred2 = [p for p in result2["predictions"] if p.get("class") == "wheelchair"]

        return pred1 + pred2
    except Exception as e:
        print(f"[ERROR] 분석 실패: {e}")
        return []

# =========================
# 3. 모델 분석 스레드
# =========================
def detection_worker(frame_queue, result_queue, stop_event):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        detections = analyze_frame(frame)

        # 결과 큐에 최신 결과만 유지
        if not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        result_queue.put(detections)

# =========================
# 4. 비디오 실행 (실시간)
# =========================
def run_video(video_path, resize_width=640):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 비디오 파일 열기 실패")
        return

    frame_queue = queue.Queue(maxsize=1)   # 최신 프레임 1개만 유지
    result_queue = queue.Queue(maxsize=1)  # 최신 탐지 결과 1개만 유지
    stop_event = threading.Event()

    # 백그라운드 스레드 실행
    worker = threading.Thread(target=detection_worker, args=(frame_queue, result_queue, stop_event), daemon=True)
    worker.start()

    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 리사이즈
        h, w = frame.shape[:2]
        scale = resize_width / w
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (resize_width, new_h))

        # 최신 프레임 큐에 넣기
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame_resized.copy())

        # 최신 탐지 결과 가져오기
        if not result_queue.empty():
            last_detections = result_queue.get()

        # 탐지 결과 화면에 표시
        for det in last_detections:
            x, y, w, h = int(det["x"]), int(det["y"]), int(det["width"]), int(det["height"])
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, "Wheelchair", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- 콘솔 출력: 탐지 여부 ---
        if last_detections:
            print(f"[DETECTED] {len(last_detections)} wheelchair(s) detected")
        else:
            print("[DETECTED] None")

        # 화면 출력
        cv2.imshow("Wheelchair Detection", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 종료 처리
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()

# =========================
# 5. 실행
# =========================
if __name__ == "__main__":
    video_file = r"../../assets/wheelchair/test/images/2025_08_21 16_30.mp4"
    run_video(video_file)