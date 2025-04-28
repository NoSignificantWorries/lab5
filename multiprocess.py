import time
import multiprocessing

# selfmade modules
import main as base

SENTINEL = object()


def cam_process(camera_descriptor, resolution, frames, stop_flag, finish_video, num_workers):
    sensor = base.SensorCam(camera_descriptor, resolution)
    next_id = 0
    while not stop_flag.is_set():
        data = sensor.get()
        print(f"Frame: {next_id}/{sensor.total_frames}", data is None)
        if data is not None:
            frames.put((next_id, data))
            next_id += 1
        else:
            finish_video.set()
            stop_flag.set()
            break
    sensor.__del__()


def detector_process(frames, stop_flag, finish_video):
    detector = base.Detector(1)
    total_time = 0.0
    n = 0
    while not stop_flag.is_set():
        try:
            frame_id, frame = frames.get(timeout=1)
            if frame is SENTINEL:
                print("No more frames. Exiting detector loop.")
                break
            print(f"Detector process got frame {frame_id}")
            start = time.time()
            id, frame = detector.predict(frame_id, frame)
            end = time.time()
            total_time += end - start
            n += 1
        except Exception:
            if finish_video.is_set():
                print("Finish event set, exiting detector loop.")
                break
            pass
    print(f"--> Total time per predict: {(total_time / (n + 1e-6)):.3f}")


def main():
    camera_descriptor = "data/output.mp4"
    resolution = (640, 480)
    num_workers = 4

    frames = multiprocessing.Queue(maxsize=num_workers * 2)
    stop_flag = multiprocessing.Event()
    finish_video = multiprocessing.Event()

    camera_process = multiprocessing.Process(
        target=cam_process,
        args=(camera_descriptor, resolution, frames, stop_flag, finish_video, num_workers)
    )
    camera_process.start()
    
    detector_processes = []
    for _ in range(num_workers):
        detector_process_object = multiprocessing.Process(
            target=detector_process,
            args=(frames, stop_flag, finish_video)
        )
        detector_processes.append(detector_process_object)
        detector_process_object.start()

    start = time.time()
    while not stop_flag.is_set():
        try:
            if finish_video.is_set() and frames.empty():
                stop_flag.set()
                print("Finish event set, exiting main loop.")
                break
        except Exception:
            if finish_video.is_set():
                print("Finish event set, exiting main loop.")
                break
    end = time.time()

    time.sleep(3)
    
    if not frames.empty():
        while not frames.empty():
            _ = frames.get()
    
    frames.close()

    print(f"==> Total time: {(end - start):.3f}")

    for detector_process_object in detector_processes:
        detector_process_object.join()
    print("Detectors closed")
    camera_process.join()


if __name__ == "__main__":
    main()

# 1 thread in torch
# 1: Total time: 285.298 (per frame: ~0.748) 1.000
# 2: Total time: 148.447 (per frame: ~0.782) 1.922
# 4: Total time: 101.814 (per frame: ~1.125) 2.802
# 8: Total time: 83.763 (per frame: ~1.844)  3.406
# 16: Total time: 79.284 (per frame: ~3.652) 3.589
# 20: Total time: 78.757 (per frame: ~4.695) 3.623

# 4 threads in torch
# 1: Total time: 161.120 (per frame: ~0.421)
# 2: Total time: 135.018 (per frame: ~0.715)
# 3: Total time: 175.160 (per frame: ~1.385)
# 4: Total time: 236.461 (per frame: ~2.500)

# 8 threads in torch
# 4: Total time: 519.508 (per frame: ~5.535)
