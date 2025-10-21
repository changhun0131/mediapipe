import cv2
import mediapipe as mp

# Mediapipe 얼굴 감지 모델 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 얼굴 감지 클래스 초기화
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# 비디오 파일 경로
input_video = "face.mp4"  # 사용할 비디오 파일 경로

# 비디오 파일 열기
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 색상 공간 변경 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 감지 수행
    results = face_detection.process(rgb_frame)

    # 감지된 얼굴에 대해 사각형 그리기
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow('Face Detection', frame)

    # 종료 조건 (ESC 키)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 비디오 파일과 윈도우 리소스 해제
cap.release()
cv2.destroyAllWindows()
