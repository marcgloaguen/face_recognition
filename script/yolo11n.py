import cv2
from ultralytics import YOLO
import torch

# Assurez-vous que MPS est disponible
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = YOLO("model/yolo11n.pt")
video_capture = cv2.VideoCapture(0)


while video_capture.isOpened():
    success, frame = video_capture.read()

    if success:
        results = model.track(frame, persist=True,device='mps',  verbose=False, line_width=4)
        annotated_frame = results[0].plot()
        cv2.imshow("WebCam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

video_capture.release()
cv2.destroyAllWindows()