import cv2
from collections import deque
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

# ===== 설정 =====
SMOOTHING_FRAMES = 5
SHOULDER_MARGIN = 0.05   # 손 들기: 어깨보다 위에 있어야 함
X_MARGIN = 0.25           # 이마 판정: 좌우 폭
Y_MARGIN = 0.12           # 이마 판정: 상하 폭

MSG_RAISE = "저 부르셨어요?"
MSG_FOREHEAD = "시원한 음료 필요하신가요?"
MSG_NONE = "동작 없음"

# 한글 폰트 자동 탐색
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
    raise FileNotFoundError("한글 폰트를 찾을 수 없습니다. 시스템에 설치 후 경로를 지정하세요.")
FONT_SIZE = 38
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# ===== MediaPipe 설정 =====
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# 상태 저장
hand_raise_queue = deque(maxlen=SMOOTHING_FRAMES)
forehead_queue = deque(maxlen=SMOOTHING_FRAMES)

# ===== 보조 함수 =====
def lm(lms, k):
    return lms[k.value]

def get_forehead_point(lms):
    nose = lm(lms, mp_pose.PoseLandmark.NOSE)
    le = lm(lms, mp_pose.PoseLandmark.LEFT_EYE)
    re = lm(lms, mp_pose.PoseLandmark.RIGHT_EYE)
    return ( (nose.x + le.x + re.x)/3, (nose.y + le.y + re.y)/3 )

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

def draw_korean_text_bgr(frame, text, xy, color_bgr):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    b, g, r = color_bgr
    draw.text(xy, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ===== 메인 루프 =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        hand_raise_state = forehead_state = False

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            forehead_state = is_hand_on_forehead(lms)
            hand_raise_state = is_hand_raised(lms) if not forehead_state else False
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        hand_raise_queue.append(hand_raise_state)
        forehead_queue.append(forehead_state)

        hand_raise_detected = len(hand_raise_queue) == hand_raise_queue.maxlen and all(hand_raise_queue)
        forehead_detected = len(forehead_queue) == forehead_queue.maxlen and all(forehead_queue)

        # 우선순위: 이마 터치 > 손 들기
        if forehead_detected:
            text, color = MSG_FOREHEAD, (255, 0, 0)
        elif hand_raise_detected:
            text, color = MSG_RAISE, (0, 255, 0)
        else:
            text, color = MSG_NONE, (0, 0, 255)

        frame = draw_korean_text_bgr(frame, text, (20, 40), color)
        cv2.imshow("Hand Gesture Detection (Korean)", frame)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
