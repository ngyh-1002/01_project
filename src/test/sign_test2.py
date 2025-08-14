import cv2
import time
from collections import deque
import mediapipe as mp
import math

# ===== 설정값 =====
SMOOTHING_FRAMES = 5
X_MARGIN = 0.25                  # 좌우 폭 (기존보다 넓게)
Y_MARGIN = 0.12                  # 위아래 범위 (y 기준)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

left_queue = deque(maxlen=SMOOTHING_FRAMES)
right_queue = deque(maxlen=SMOOTHING_FRAMES)

def get_forehead_point(lms):
    nose = lms[mp_pose.PoseLandmark.NOSE]
    left_eye = lms[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = lms[mp_pose.PoseLandmark.RIGHT_EYE]

    forehead_x = (nose.x + left_eye.x + right_eye.x) / 3
    forehead_y = (nose.y + left_eye.y + right_eye.y) / 3
    return type('P', (object,), {'x': forehead_x, 'y': forehead_y})()

def is_hand_on_forehead(lms, side="left"):
    forehead = get_forehead_point(lms)
    if side == "left":
        wrist = lms[mp_pose.PoseLandmark.LEFT_WRIST]
    else:
        wrist = lms[mp_pose.PoseLandmark.RIGHT_WRIST]

    if wrist.visibility < 0.5:
        return False

    # y는 그대로 두고, x는 폭을 넓게
    x_condition = abs(wrist.x - forehead.x) <= X_MARGIN
    y_condition = abs(wrist.y - forehead.y) <= Y_MARGIN

    return x_condition and y_condition

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        left_state = right_state = False

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            left_state = is_hand_on_forehead(lms, "left")
            right_state = is_hand_on_forehead(lms, "right")
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        left_queue.append(left_state)
        right_queue.append(right_state)

        left_forehead = len(left_queue) == left_queue.maxlen and all(left_queue)
        right_forehead = len(right_queue) == right_queue.maxlen and all(right_queue)

        status_text = []
        if left_forehead:
            status_text.append("LEFT HAND ON FOREHEAD")
        if right_forehead:
            status_text.append("RIGHT HAND ON FOREHEAD")
        if not status_text:
            status_text.append("NO HAND ON FOREHEAD")

        cv2.putText(frame, " | ".join(status_text), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if "FOREHEAD" in status_text[0] else (0, 0, 255), 2)

        cv2.imshow("Forehead Touch Detection", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
