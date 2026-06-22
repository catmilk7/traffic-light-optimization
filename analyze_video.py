import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import time

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

def pick_video():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select your video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )

def get_screen_size():
    """Grab the screen resolution so the preview window never exceeds it."""
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h

def count_vehicles(video_path, show_preview=True, confidence=0.25, max_preview_fraction=0.6):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video: {total_frames} frames at {fps:.1f} fps ({total_frames/fps:.0f}s)")
    print(f"Processing...\n")

    # Work out a preview size that comfortably fits on screen
    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * max_preview_fraction)
    max_h = int(screen_h * max_preview_fraction)

    # Load the YOLOv8 model (will download if not present)
    model = YOLO("yolov8s.pt")

    results = model(
        source=video_path,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=confidence,
        stream=True,
        show=False,   # IMPORTANT: don't let ultralytics open its own full-size window
        verbose=False,
    )

    frame_stats = []
    start_time = time.time()
    window_ready = False

    for frame_idx, r in enumerate(results):
        counts = {label: 0 for label in VEHICLE_CLASSES.values()}

        for box in r.boxes:
            cls_id = int(box.cls)
            label = VEHICLE_CLASSES.get(cls_id)
            if label:
                counts[label] += 1

        total = sum(counts.values())
        frame_stats.append({"frame": frame_idx + 1, "total": total, **counts})

        if show_preview:
            annotated = r.plot()
            h, w = annotated.shape[:2]
            # Scale to fit within max_w x max_h, keeping aspect ratio
            scale = min(max_w / w, max_h / h, 1.0)
            preview = cv2.resize(annotated, (int(w * scale), int(h * scale)))

            if not window_ready:
                cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Vehicle Detection", preview.shape[1], preview.shape[0])
                window_ready = True

            cv2.imshow("Vehicle Detection", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if (frame_idx + 1) % 30 == 0 or (frame_idx + 1) == total_frames:
            elapsed = time.time() - start_time
            frames_done = frame_idx + 1
            rate = frames_done / elapsed                        # frames/sec
            remaining = (total_frames - frames_done) / rate    # seconds left
            pct = frames_done / total_frames * 100
            bar = ("█" * int(pct // 5)).ljust(20)
            eta = time.strftime("%M:%S", time.gmtime(remaining))
            print(f"\r[{bar}] {pct:5.1f}%  frame {frames_done}/{total_frames}  |  {rate:.1f} fps  |  ETA {eta}  |  now: {total} vehicles", end="", flush=True)

    if show_preview:
        cv2.destroyAllWindows()

    elapsed_total = time.time() - start_time
    print(f"\nDone in {time.strftime('%M:%S', time.gmtime(elapsed_total))}.\n")
    return frame_stats

if __name__ == "__main__":
    path = pick_video()
    if not path:
        print("No file selected.")
    else:
        print(f"Running YOLO on: {path}\n")
        stats = count_vehicles(path, show_preview=True, max_preview_fraction=0.6)

        totals = [s["total"] for s in stats]
        print(f"── Summary ──────────────────────")
        print(f"Frames analysed : {len(stats)}")
        print(f"Avg vehicles    : {sum(totals)/len(totals):.1f} per frame")
        print(f"Peak frame      : {max(totals)} vehicles")

        for label in ("car", "motorcycle", "bus", "truck"):
            avg = sum(s[label] for s in stats) / len(stats)
            print(f"  avg {label:<12}: {avg:.1f}/frame")