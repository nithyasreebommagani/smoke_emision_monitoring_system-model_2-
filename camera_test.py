import cv2

for i in range(5):
    print(f"\nTesting Camera {i}")

    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)

    print("Opened:", cap.isOpened())

    if cap.isOpened():
        ret, frame = cap.read()
        print("Frame Read:", ret)

    cap.release()