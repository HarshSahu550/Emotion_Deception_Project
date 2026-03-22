import cv2

from face_module.face_detector import detect_faces


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        emotions = emotion_module.emotion_inference.predict_emotion(face)

        emotion = max(emotions, key=emotions.get)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()