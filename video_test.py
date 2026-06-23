import cv2

video_path = r"C:\Users\E028.26\Downloads\WhatsApp Video 2026-06-21 at 12.31.03 PM.mp4"

cap = cv2.VideoCapture(video_path)

print("Opened:", cap.isOpened())

ret, frame = cap.read()

print("Read:", ret)

if ret:
    print(frame.shape)

cap.release()