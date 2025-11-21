import  cv2
import os
from ultralytics import YOLO
from deepface import DeepFace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU for DeepFace
cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')
while True:
    success, frame = cap.read()
   

    if not success:
        print("Failed to grab frame")
        break
    
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = frame[y1:y2, x1:x2]
            analysis = DeepFace.analyze(face_crop
                                , actions = ['age','emotion']
                                , enforce_detection = False)
            my_age = analysis[0]['age']
            my_emotion = analysis[0]['dominant_emotion']
            cv2.putText(frame, f'Age: {my_age}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Emotion: {my_emotion}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            result = result[0].plot()
            cv2.imshow("camera", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()