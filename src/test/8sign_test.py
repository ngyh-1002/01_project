import cv2
import threading
import queue
import os
import numpy as np
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# ===== 설정 =====
SMOOTHING_FRAMES = 5
SHOULDER_MARGIN = 0.05
X_MARGIN = 0.25
Y_MARGIN = 0.12
MAX_PEOPLE = 5

MSG_RAISE = "저 부르셨어요?"
MSG_FOREHEAD = "시원한 음료 필요하신가요?"
MSG_NONE = "동작 없음"
MSG_BLOW = "입김 부는 중"

# ===== 한글 폰트 설정 =====
def find_korean_font():
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",  # Windows
        "/System/Library/Fonts/AppleGothic.ttf",  # macOS
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_korean_font()
if FONT_PATH is None:
    raise FileNotFoundError("한글 폰트를 찾을 수 없습니다.")
FONT_SIZE = 28
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

def draw_korean_text_bgr(frame, text, xy, color_bgr):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    b, g, r = color_bgr
    draw.text(xy, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ===== YOLOv8 모델 (GPU) =====
yolo_model = YOLO("yolov8n.pt").to("cuda")

# ===== MediaPipe Pose (CPU) =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ===== 사람별 상태 저장 =====
class PersonROI:
    def __init__(self):
        self.hand_raise_queue = deque(maxlen=SMOOTHING_FRAMES)
        self.forehead_queue = deque(maxlen=SMOOTHING_FRAMES)
        self.blow_queue = deque(maxlen=SMOOTHING_FRAMES)

# ===== ROI별 상태 확인 함수 =====
def lm(lms, k):
    return lms[k.value]

def get_forehead_point(lms):
    nose = lm(lms, mp_pose.PoseLandmark.NOSE)
    le = lm(lms, mp_pose.PoseLandmark.LEFT_EYE)
    re = lm(lms, mp_pose.PoseLandmark.RIGHT_EYE)
    return ((nose.x + le.x + re.x)/3, (nose.y + le.y + re.y)/3)

def is_hand_raised(lms):
    lw = lm(lms, mp_pose.PoseLandmark.LEFT_WRIST)
    rw = lm(lms, mp_pose.PoseLandmark.RIGHT_WRIST)
    ls = lm(lms, mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs = lm(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_up = lw.visibility > 0.5 and lw.y < ls.y - SHOULDER_MARGIN
    right_up = rw.visibility > 0.5 and rw.y < rs.y - SHOULDER_MARGIN
    return left_up or right_up

def is_hand_on_forehead(lms):
    fx, fy = get_forehead_point(lms)
    lw = lm(lms, mp_pose.PoseLandmark.LEFT_WRIST)
    rw = lm(lms, mp_pose.PoseLandmark.RIGHT_WRIST)
    def near(w):
        if w.visibility < 0.5:
            return False
        x_condition = abs(w.x - fx) <= X_MARGIN
        y_condition = abs(w.y - fy) <= Y_MARGIN
        return x_condition and y_condition
    return near(lw) or near(rw)

def is_blowing(lms):
    lw = lm(lms, mp_pose.PoseLandmark.LEFT_WRIST)
    rw = lm(lms, mp_pose.PoseLandmark.RIGHT_WRIST)
    nose = lm(lms, mp_pose.PoseLandmark.NOSE)

    if lw.visibility < 0.5 or rw.visibility < 0.5:
        return False

    y_offset = 0.05
    mouth_y = nose.y + y_offset
    mouth_x = nose.x

    X_THRESHOLD = 0.18
    BASE_Y_THRESHOLD = 0.22
    EXTRA_Y_THRESHOLD = 0.18

    def near_mouth(w):
        x_condition = abs(w.x - mouth_x) <= X_THRESHOLD
        if w.y > nose.y:
            y_threshold = BASE_Y_THRESHOLD + EXTRA_Y_THRESHOLD
        else:
            y_threshold = BASE_Y_THRESHOLD
        y_condition = mouth_y - 0.02 <= w.y <= mouth_y + y_threshold
        return x_condition and y_condition

    return near_mouth(lw) and near_mouth(rw)

# ===== 스레드 통신 큐 =====
frame_queue = queue.Queue(maxsize=2)
yolo_queue = queue.Queue(maxsize=2)
pose_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

# ===== YOLO 스레드 (GPU) =====
def yolo_thread():
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        results = yolo_model(frame, device="cuda")[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        person_boxes = [
            box for box, cls_id, conf in zip(boxes, class_ids, confidences)
            if cls_id == 0 and conf > 0.5
        ][:MAX_PEOPLE]
        yolo_queue.put((frame, person_boxes))

# ===== Pose 스레드 (CPU) =====
def pose_thread():
    person_rois = []
    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:
        while not stop_event.is_set():
            try:
                frame, boxes = yolo_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # ROI 객체 수 맞추기
            while len(person_rois) < len(boxes):
                person_rois.append(PersonROI())

            annotated = frame.copy()

            for roi_obj, (x1, y1, x2, y2) in zip(person_rois, boxes):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result = pose.process(img_rgb)

                hand_raise_state = forehead_state = blow_state = False

                if result.pose_landmarks:
                    lms = result.pose_landmarks.landmark
                    forehead_state = is_hand_on_forehead(lms)
                    hand_raise_state = is_hand_raised(lms) if not forehead_state else False
                    blow_state = is_blowing(lms) if not (forehead_state or hand_raise_state) else False

                    # ROI → 원본 좌표 변환
                    for lm_point in lms:
                        lm_point.x = lm_point.x * (x2 - x1) + x1
                        lm_point.y = lm_point.y * (y2 - y1) + y1

                    mp_drawing.draw_landmarks(annotated, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 상태 저장
                roi_obj.hand_raise_queue.append(hand_raise_state)
                roi_obj.forehead_queue.append(forehead_state)
                roi_obj.blow_queue.append(blow_state)

                hand_raise_detected = len(roi_obj.hand_raise_queue) == SMOOTHING_FRAMES and all(roi_obj.hand_raise_queue)
                forehead_detected = len(roi_obj.forehead_queue) == SMOOTHING_FRAMES and all(roi_obj.forehead_queue)
                blow_detected = len(roi_obj.blow_queue) == SMOOTHING_FRAMES and all(roi_obj.blow_queue)

                if forehead_detected:
                    text, color = MSG_FOREHEAD, (255, 0, 0)
                elif hand_raise_detected:
                    text, color = MSG_RAISE, (0, 255, 0)
                elif blow_detected:
                    text, color = MSG_BLOW, (0, 255, 255)
                else:
                    text, color = MSG_NONE, (0, 0, 255)

                annotated = draw_korean_text_bgr(annotated, text, (x1, y1 - 20), color)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            pose_queue.put(annotated)

# ===== 메인 루프 =====
cap = cv2.VideoCapture(0)

# 스레드 시작
t1 = threading.Thread(target=yolo_thread, daemon=True)
t2 = threading.Thread(target=pose_thread, daemon=True)
t1.start()
t2.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not frame_queue.full():
            frame_queue.put(frame)

        if not pose_queue.empty():
            display_frame = pose_queue.get()
            cv2.imshow("YOLO-Pose Multi-Person Detection (Threaded)", display_frame)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

finally:
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
