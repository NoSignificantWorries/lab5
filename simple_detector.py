import time

import cv2

import main as base


def main():
    cam = base.SensorCam("data/output.mp4", (640, 480))
    detector = base.Detector(64)
    
    time.sleep(2)
    start = time.time()
    total_time = 0.0
    n = 0
    while True:
        frame = cam.get()
        if frame is None:
            break
        start_predict = time.time()
        _, frame = detector.predict(0, frame)
        end_predict = time.time()
        total_time += end_predict - start_predict
        n += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end = time.time()
    print(f"Time taken: {(end - start):.2f} seconds. Time per frame: {(total_time / n):.3f} seconds.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# 1: Time taken: 285.55 seconds. Time per frame: 0.740 seconds.
# 2: Time taken: 145.81 seconds. Time per frame: 0.379 seconds.
# 4: Time taken: 88.00 seconds. Time per frame: 0.229 seconds.
# 6: Time taken: 120.19 seconds. Time per frame: 0.312 seconds.
# 8: Time taken: 101.57 seconds. Time per frame: 0.264 seconds.
# 16: Time taken: 109.84 seconds. Time per frame: 0.285 seconds.
# 32: Time taken: 116.07 seconds. Time per frame: 0.302 seconds.
# 64: Time taken: 128.92 seconds. Time per frame: 0.335 seconds.
