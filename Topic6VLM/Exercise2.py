import cv2
import base64
import io
import json
from PIL import Image
import ollama


VIDEO_PATH = "video.mp4"
FRAME_INTERVAL_SECONDS = 2
MODEL_NAME = "llava"


def extract_frames(video_path, interval_seconds):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_seconds)

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


def frame_to_base64(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def ask_llava_if_person(base64_image):
    prompt = """
    Answer ONLY with JSON:
    {
      "person_present": true or false
    }

    Is there a person visible in this scene?
    """

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

    text = response["message"]["content"]

    try:
        parsed = json.loads(text)
        return parsed["person_present"]
    except:
        # fallback heuristic
        return "true" in text.lower()


def detect_entry_exit(timestamps, presence_flags):
    events = []
    previous = False

    for time_sec, present in zip(timestamps, presence_flags):
        if present and not previous:
            events.append(("ENTRY", time_sec))
        elif not present and previous:
            events.append(("EXIT", time_sec))

        previous = present

    return events


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def main():
    print("Extracting frames...")
    frames, timestamps = extract_frames(VIDEO_PATH, FRAME_INTERVAL_SECONDS)

    print(f"{len(frames)} frames extracted.")

    presence_flags = []

    for i, frame in enumerate(frames):
        print(f"Analyzing frame {i+1}/{len(frames)}...")
        b64 = frame_to_base64(frame)
        person_present = ask_llava_if_person(b64)
        presence_flags.append(person_present)

    events = detect_entry_exit(timestamps, presence_flags)

    print("\n--- Surveillance Report ---\n")

    if not events:
        print("No person detected in video.")
        return

    for event_type, time_sec in events:
        print(f"{event_type} at {format_time(time_sec)}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
