import torch
import cv2
import numpy as np

model = torch.hub.load("yolov5", 'custom', r"C:\Users\rohin\Desktop\pythonProject\yolov5n.pt", source='local')

cap = cv2.VideoCapture(r"C:\Users\rohin\Desktop\pythonProject\Best Dunks Of The 2021-22 NBA Season 🔥🔥.mp4")

while True:
    ret ,frame =cap.read()

    results= model(frame)

    cv2.imshow("frame", np.squeeze(results.render()))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()

# Closes all the frames
cv2.destroyAllWindows()