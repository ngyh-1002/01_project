import cv2
from collections import deque
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from ultralytics import YOLO
import os

# ===== 설정 =====
SMOOTHING_FRAMES = 5
SHOULDER_MARGIN = 0.05
X_MARGIN = 0.25
Y_MARGIN = 0.12

MSG_RAISE = "저 부르셨어요?"
MSG_FOREHEAD = "시원한 음료 필요하신가요?"
MSG_NONE = "동작 없음"
MSG_BLOW = "입김 부는 중"

MAX_PEOPLE = 5  # 실시간 처리용 최대 사람 수

# 한글 폰트
def find_korean_font():
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
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

# ===== YOLOv8 모델 로드 =====
yolo_model = YOLO("yolov8n.pt")  # 사람 탐지 전용 YOLO 모델

# ===== MediaPipe Pose =====
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 사람별 MediaPipe Pose 객체와 상태 저장
class PersonROI:
    def __init__(self):
        self.pose = mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.hand_raise_queue = deque(maxlen=SMOOTHING_FRAMES)
        self.forehead_queue = deque(maxlen=SMOOTHING_FRAMES)
        self.blow_queue = deque(maxlen=SMOOTHING_FRAMES)

# ROI별 상태 확인 함수
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
    X_THRESHOLD = 0.12
    Y_THRESHOLD = 0.18
    def near_mouth(w):
        x_condition = abs(w.x - mouth_x) <= X_THRESHOLD
        y_condition = mouth_y - 0.02 <= w.y <= mouth_y + Y_THRESHOLD
        return x_condition and y_condition
    return near_mouth(lw) and near_mouth(rw)

# ===== 메인 루프 =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

person_rois = []  # PersonROI 리스트

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)[0]  # YOLO 탐지
        boxes = results.boxes.xyxy.cpu().numpy()  # 탐지된 박스
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        # 사람만 필터
        person_boxes = []
        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            if cls_id == 0 and conf > 0.5:  # 0: person class
                person_boxes.append(box)
        person_boxes = person_boxes[:MAX_PEOPLE]

        # ROI 수와 PersonROI 수 맞추기
        while len(person_rois) < len(person_boxes):
            person_rois.append(PersonROI())

        display_frame = frame.copy()

        for roi_obj, box in zip(person_rois, person_boxes):
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            result = roi_obj.pose.process(img_rgb)

            hand_raise_state = forehead_state = blow_state = False

            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                forehead_state = is_hand_on_forehead(lms)
                hand_raise_state = is_hand_raised(lms) if not forehead_state else False
                blow_state = is_blowing(lms) if not (forehead_state or hand_raise_state) else False

                # ROI 좌표 → 원본 프레임
                for lm_point in lms:
                    lm_point.x = lm_point.x * (x2 - x1) + x1
                    lm_point.y = lm_point.y * (y2 - y1) + y1

                mp_drawing.draw_landmarks(display_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 상태 업데이트
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

            display_frame = draw_korean_text_bgr(display_frame, text, (x1, y1-20), color)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("YOLO-Pose Multi-Person Detection", display_frame)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    for roi in person_rois:
        roi.pose.close()
