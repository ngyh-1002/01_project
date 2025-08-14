import cv2
import mediapipe as mp
import os

# ===== 설정 =====
SHOULDER_MARGIN = 0.05
X_MARGIN = 0.25
Y_MARGIN = 0.12

MSG_RAISE = "저 부르셨어요?"
MSG_FOREHEAD = "시원한 음료 필요하신가요?"
MSG_NONE = "동작 없음"

# ===== MediaPipe 설정 =====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

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

# ===== 한 장 이미지 처리 =====
IMAGE_PATH = "./callme.jpg"  # 테스트할 이미지 파일

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"{IMAGE_PATH} 읽기 실패")

img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
result = pose.process(img_rgb)

if result.pose_landmarks:
    lms = result.pose_landmarks.landmark
    forehead_state = is_hand_on_forehead(lms)
    hand_raise_state = is_hand_raised(lms) if not forehead_state else False
else:
    forehead_state = hand_raise_state = False

# 우선순위: 이마 터치 > 손 들기
if forehead_state:
    text = MSG_FOREHEAD
elif hand_raise_state:
    text = MSG_RAISE
else:
    text = MSG_NONE

print(f"이미지 {IMAGE_PATH} 결과: {text}")

pose.close()
