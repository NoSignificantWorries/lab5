import os
import time
import queue
import threading
import logging
import argparse

import numpy as np
import cv2
import ultralytics as ult


class Global:
    SKELETON = [
            (3, 1), (1, 0), (0, 2), (2, 4), (1, 2), (4, 6), (3,5),
            (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10),
            (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16),
            (5, 11), (6, 12)
    ]

    COLORS = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170)
    ]
    num_workers = 12
    buffer_size = 2
    input_frame_buffer = queue.Queue(maxsize=buffer_size)
    output_frame_buffer = {}
    output_frame_buffer_lock = threading.Lock()
    stop_flag = threading.Event()
    finish_video = threading.Event()
    next_id = 0


def draw_pose(frame: np.ndarray, keypoints: np.ndarray, confs: np.ndarray,
              skeleton: list[tuple[int, int]] = Global.SKELETON, colors: list[tuple[int, int, int]] = Global.COLORS, kp_thresh: float = 0.5) -> None:
    for idx, ((x, y), conf) in enumerate(zip(keypoints, confs)):
        if conf > kp_thresh:
            cv2.circle(frame, (int(x), int(y)), 4, colors[idx % len(colors)], -1)
    
    for i, (start, end) in enumerate(skeleton):
        if confs[start] > kp_thresh and confs[end] > kp_thresh:
            pt1 = tuple(map(int, keypoints[start]))
            pt2 = tuple(map(int, keypoints[end]))
            cv2.line(frame, pt1, pt2, colors[i % len(colors)], 2)


class Detector:
    def __init__(self):
        self.model = ult.YOLO("yolov8s-pose.pt")
    
    def predict(self, id: int, frame: np.ndarray) -> tuple[int, np.ndarray]:
        results = self.model(frame)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            keypoints = result.keypoints.cpu().numpy()

            for box, keypoint in zip(boxes, keypoints):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {(box.conf[0]):.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                draw_pose(frame, keypoint.xy[0], keypoint.conf[0])

        return id, frame



class SensorCam:
    def __init__(self, camera_descriptor: str, resolution: tuple[int, int]):
        self._resolution = resolution
        self.cap = cv2.VideoCapture(camera_descriptor)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video device: {camera_descriptor}")

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        resized_frame = cv2.resize(frame, self._resolution, cv2.INTER_CUBIC)
        return resized_frame


class WindowImage:
    def __init__(self, resolution: tuple[int, int], show: bool = True, save: bool = False, out_file: str = "results/res.mp4"):
        self._resolution = resolution
        self._save = save
        self._show = show
        if save:
            self.out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, self._resolution)
        else:
            self.out = None

    def __del__(self):
        if self._save:
            self.out.release()
        cv2.destroyAllWindows()

    def show(self, frame: np.ndarray):
        if frame is None:
            raise RuntimeError("Unknown error occured. Can`t read new frame!")
        if self._show:
            cv2.imshow("Cam", frame)
        if self._save:
            self.out.write(frame)


def DetectorThread(detector: Detector):
    while not Global.stop_flag.is_set():
        try:
            id, frame = Global.input_frame_buffer.get(timeout=0.1)
        except queue.Empty:
            continue
        id, frame = detector.predict(id, frame)
        with Global.output_frame_buffer_lock:
            Global.output_frame_buffer[id] = frame


def CamThread(sensor: SensorCam):
    while not Global.stop_flag.is_set():
        data = sensor.get()
        print(f"Frame: {Global.next_id}/{sensor.total_frames}", data is None)
        if data is not None:
            Global.input_frame_buffer.put((Global.next_id, data))
            Global.next_id += 1
        else:
            Global.finish_video.set()
            break


def main(input_file: str, output_file: str, resolution: str) -> None:
    if not os.path.exists("log"):
        os.makedirs("log")
    
    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger("my_logger")

    try:
        resolution = tuple(map(int, resolution.split("x")))
    except BaseException:
        logger.critical("Wrong resolution format, it should be <integer>x<integer>!")
        raise ValueError("Wrong resolution format.")
    
    if resolution[0] <= 0 or resolution[1] <= 0:
        logger.critical(f"Wrong dimension values width: {resolution[0]}, height: {resolution[1]}")
        raise ValueError("Wrong resolution format.")

    try:
        cam = SensorCam(input_file, resolution)
    except ValueError as err:
        logger.critical(err)
        raise ValueError("Errors with camera sensor.")
    
    cam_thread = threading.Thread(target=CamThread, args=(cam,))
    cam_thread.daemon = True
    cam_thread.start()
    
    time.sleep(0.5)
    detectors = []
    detector_threads = []
    for _ in range(Global.num_workers):
        detector = Detector()
        thread = threading.Thread(target=DetectorThread, args=(detector,))
        thread.daemon = True
        thread.start()
        detectors.append(detector)
        detector_threads.append(thread)
    
    window = WindowImage(resolution, False, output_file is not None, output_file)
    
    try:
        last_id = 0
        time.sleep(1)
        start = time.time()
        while not Global.stop_flag.is_set():
            if Global.finish_video.is_set() and Global.input_frame_buffer.empty():
                Global.stop_flag.set()
                break
            try:
                with Global.output_frame_buffer_lock:
                    if len(Global.output_frame_buffer.keys()) <= 0:
                        continue
                    if max(Global.output_frame_buffer.keys()) <= Global.buffer_size:
                        continue
                    if last_id in Global.output_frame_buffer.keys():
                        try:
                            window.show(Global.output_frame_buffer[last_id])
                            del Global.output_frame_buffer[last_id]
                            last_id += 1
                        except RuntimeError as err:
                            logger.critical(err)
                            Global.stop_flag.set()
                            raise RuntimeError("Unknown errors with camera.")
            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                Global.stop_flag.set()
                break

    except KeyboardInterrupt:
        Global.stop_flag.set()
        print("Exit...")

    end = time.time()
    time.sleep(2)

    print(f"Total time: {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_file", type=str, help="Name of your video device", default="/dev/video0")
    parser.add_argument("-o", "--output_file", type=str, help="Name of your video device", default=None)
    parser.add_argument("-r", "--resolution", type=str, help="Video resolution", default="640x480")

    args = parser.parse_args()
    
    try:
        main(args.input_file, args.output_file, args.resolution)
    except BaseException as exp:
        print(f"Some errors occured: {exp}")
