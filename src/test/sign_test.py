import cv2
import time
from collections import deque
import mediapipe as mp

# ===== 설정값 =====
SMOOTHING_FRAMES = 5       # 연속으로 조건을 만족해야 하는 프레임 수
SHOULDER_MARGIN = 0.02     # 어깨 대비 손목이 이 정도 이상 위에 있어야 "들었다"로 간주 (정규화 좌표)
ELBOW_ABOVE_SHOULDER_OK = True  # 팔꿈치도 어깨 근처 이상으로 올라오면 더 확실하게 판단
DRAW_TEXT = True

# ===== MediaPipe 초기화 =====
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===== 상태 버퍼(연속 프레임 판정용) =====
left_queue = deque(maxlen=SMOOTHING_FRAMES)
right_queue = deque(maxlen=SMOOTHING_FRAMES)

def is_hand_raised(landmarks, side="left"):
    """
    landmarks: pose_landmarks.landmark
    side: "left" or "right"
    판단 기준:
      - 손목(wrist).y < 어깨(shoulder).y - margin  → 어깨보다 위
      - (선택) 팔꿈치(elbow).y도 어깨 근처 이상으로 올라오면 더 확실
    """
    if side == "left":
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    else:
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # MediaPipe 좌표계: y는 위가 0, 아래로 갈수록 값이 커짐 → "위"는 값이 더 작음
    wrist_above = wrist.visibility > 0.5 and (wrist.y < (shoulder.y - SHOULDER_MARGIN))

    if not ELBOW_ABOVE_SHOULDER_OK:
        return wrist_above

    elbow_ok = elbow.visibility > 0.5 and (elbow.y < (shoulder.y + 0.05))  # 어깨보다 조금 아래까지 허용
    return wrist_above and elbow_ok

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 다른 장치 번호(1, 2...)를 시도하세요.")

prev_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        left_state = False
        right_state = False

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark

            # 왼손/오른손 판정
            left_state = is_hand_raised(lms, "left")
            right_state = is_hand_raised(lms, "right")

            # 그림 그리기
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
            )

        # 스무딩(연속 프레임)
        left_queue.append(left_state)
        right_queue.append(right_state)

        left_raised = len(left_queue) == left_queue.maxlen and all(left_queue)
        right_raised = len(right_queue) == right_queue.maxlen and all(right_queue)

        # 텍스트 표시
        if DRAW_TEXT:
            h, w = frame.shape[:2]
            status = []
            if left_raised:
                status.append("LEFT HAND RAISED")
            if right_raised:
                status.append("RIGHT HAND RAISED")
            if not status:
                status.append("NO HAND RAISED")

            text = " | ".join(status)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if "RAISED" in text else (0, 0, 255), 2)

            # FPS
            cur_time = time.time()
            fps = 1.0 / (cur_time - prev_time) if prev_time else 0.0
            prev_time = cur_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Hand Raise Detection (MediaPipe Pose)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC 또는 q 종료
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
