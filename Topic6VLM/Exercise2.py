"""
Exercise 2 — Video-Surveillance Agent

Uses LLaVA indirectly on video by extracting frames every ~2 seconds and
asking LLaVA if a person is present in each frame. Reports the times at
which a person enters and exits the scene.
"""

import base64
import io
import json
from pathlib import Path

import cv2
import ollama
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

VIDEO_PATH = "/mnt/sdb/arafat/agentic_ai/Topic6VLM/2min.mp4"
FRAME_INTERVAL_SECONDS = 2
MODEL_NAME = "llava:latest"
FRAMES_OUTPUT_DIR = "frames"
MAX_IMAGE_DIMENSION = 512  # Reduce resolution for faster inference


# =============================================================================
# FRAME EXTRACTION
# =============================================================================

def extract_frames(video_path: str, interval_seconds: float):
    """
    Extract frames from video at the specified interval.

    Args:
        video_path: Path to the video file
        interval_seconds: Extract one frame every N seconds

    Returns:
        (frames, timestamps): List of BGR frames and their timestamps in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    interval = max(1, int(fps * interval_seconds))

    frames = []
    timestamps = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval == 0:
            time_sec = frame_num / fps
            frames.append(frame)
            timestamps.append(time_sec)
        frame_num += 1

    cap.release()
    return frames, timestamps


def frame_to_base64(frame, max_dimension: int = MAX_IMAGE_DIMENSION) -> str:
    """
    Convert an OpenCV BGR frame to base64-encoded JPEG.

    Optionally resizes to speed up inference.
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max_dimension and (img.width > max_dimension or img.height > max_dimension):
        ratio = min(max_dimension / img.width, max_dimension / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =============================================================================
# PERSON DETECTION VIA LLAVA
# =============================================================================

def ask_llava_if_person(base64_image: str) -> bool:
    """
    Ask LLaVA if a person is visible in the scene.

    Returns True if a person is present, False otherwise.
    """
    prompt = """Answer ONLY with valid JSON, no other text:
{"person_present": true}
or
{"person_present": false}

Is there a person visible in this scene?"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image],
            }
        ],
    )

    text = response.get("message", {}).get("content", "").strip()

    try:
        # Try to parse JSON from the response
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{"):
                parsed = json.loads(line)
                return bool(parsed.get("person_present", False))
        # Fallback: look for JSON in the text
        start = text.find("{")
        if start >= 0:
            end = text.rfind("}") + 1
            if end > start:
                parsed = json.loads(text[start:end])
                return bool(parsed.get("person_present", False))
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback heuristic
    return "true" in text.lower() or "yes" in text.lower() or "person" in text.lower()


# =============================================================================
# ENTRY/EXIT DETECTION
# =============================================================================

def detect_entry_exit(timestamps: list, presence_flags: list) -> list:
    """
    Detect when a person enters and exits based on presence flags.

    Returns list of (event_type, timestamp) tuples, e.g. [("ENTRY", 12.5), ("EXIT", 45.0)]
    """
    events = []
    previous = False

    for time_sec, present in zip(timestamps, presence_flags):
        if present and not previous:
            events.append(("ENTRY", time_sec))
        elif not present and previous:
            events.append(("EXIT", time_sec))
        previous = present

    return events


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"Error: Video not found: {VIDEO_PATH}")
        return

    print("=" * 60)
    print("Exercise 2 — Video-Surveillance Agent")
    print("=" * 60)
    print(f"\nVideo: {VIDEO_PATH}")
    print(f"Frame interval: {FRAME_INTERVAL_SECONDS} seconds")
    print(f"Model: {MODEL_NAME}\n")

    print("Extracting frames...")
    frames, timestamps = extract_frames(str(video_path), FRAME_INTERVAL_SECONDS)
    print(f"Extracted {len(frames)} frames.")

    # Optionally save frames to disk
    frames_dir = Path(FRAMES_OUTPUT_DIR)
    frames_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), frame)
    print(f"Saved frames to {FRAMES_OUTPUT_DIR}/\n")

    print("Analyzing frames with LLaVA...")
    presence_flags = []
    for i, frame in enumerate(frames):
        b64 = frame_to_base64(frame)
        person_present = ask_llava_if_person(b64)
        presence_flags.append(person_present)
        status = "PERSON" if person_present else "empty"
        print(f"  Frame {i + 1}/{len(frames)} ({format_time(timestamps[i])}): {status}")

    events = detect_entry_exit(timestamps, presence_flags)

    print("\n" + "=" * 60)
    print("Surveillance Report")
    print("=" * 60)

    if not events:
        print("No person detected entering or exiting the scene.")
    else:
        for event_type, time_sec in events:
            print(f"  {event_type} at {format_time(time_sec)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
