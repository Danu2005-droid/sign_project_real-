import os
import cv2
import time
import json
import subprocess
import numpy as np
import mediapipe as mp
import warnings
import queue
import threading

from collections import deque
from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# ---------------- SPEECH ----------------

speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            subprocess.run(
                ["powershell", "-Command",
                 f"Add-Type -AssemblyName System.Speech; "
                 f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                 f"$s.Speak('{text}');"],
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        except Exception as e:
            print("Speech Error:", e)
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

last_spoken      = ""
last_spoken_time = 0

def speak_text(text):
    global last_spoken, last_spoken_time
    now = time.time()
    if text == last_spoken and (now - last_spoken_time) < 3:
        return
    last_spoken      = text
    last_spoken_time = now
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except:
            pass
    speech_queue.put(text)

# ---------------- CONFIG ----------------

MODEL_PATH  = "modelnet_model.h5"
LABELS_PATH = "labels.json"
IMG_SIZE    = 224

CONFIDENCE_THRESHOLD = 0.60
SMOOTH_WINDOW        = 10
SMOOTH_MIN_VOTES     = 6
LETTER_HOLD_TIME     = 2.0
WORD_GAP_TIME        = 3.0

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

# ---------------- SHARED STATE (thread-safe) ----------------
state = {
    "prediction"      : "---",
    "confidence"      : 0.0,
    "sentence"        : "",
    "last_added"      : "",
    "last_added_time" : 0.0,
    "last_gesture_time": time.time(),
}
sentence_buffer = []
state_lock      = threading.Lock()

# ---------------- LABELS ----------------

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        class_indices = json.load(f)
    LABELS = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
else:
    LABELS = [str(i) for i in range(10)] + \
             [chr(i) for i in range(ord('A'), ord('Z')+1)]

print("Labels:", LABELS)

# ---------------- LOAD MODEL ----------------

model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded |", len(LABELS), "classes")

# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands_video = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands_image = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

# ---------------- SMOOTHING BUFFER ----------------

prediction_buffer = deque(maxlen=SMOOTH_WINDOW)

# ---------------- CORE FUNCTIONS ----------------

def extract_hand_crop(frame, mode="video"):
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = hands_image if mode == "image" else hands_video
    results  = detector.process(rgb)

    if not results.multi_hand_landmarks:
        return None, None

    h, w, _ = frame.shape
    for hand_landmarks in results.multi_hand_landmarks:
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]

        pad = 60
        x1  = max(0, int(min(xs)) - pad)
        y1  = max(0, int(min(ys)) - pad)
        x2  = min(w, int(max(xs)) + pad)
        y2  = min(h, int(max(ys)) + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        return crop, (x1, y1, x2, y2)

    return None, None


def preprocess(crop):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_frame(frame, mode="video"):
    crop, bbox = extract_hand_crop(frame, mode=mode)
    if crop is None:
        return "NO_HAND", 0.0, None

    preds      = model.predict(preprocess(crop), verbose=0)[0]
    idx        = int(np.argmax(preds))
    confidence = float(preds[idx])
    label      = LABELS[idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return "LOW_CONF", confidence, bbox

    return label, confidence, bbox


# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    npimg = np.frombuffer(request.files["file"].read(), np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"})

    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)))

    label, conf, _ = predict_frame(img, mode="image")

    if label == "NO_HAND":
        return jsonify({"prediction": "No Hand Detected", "confidence": 0})
    if label == "LOW_CONF":
        return jsonify({"prediction": "Unknown / Not Sure", "confidence": round(conf, 2)})

    speak_text(label)
    return jsonify({"prediction": label, "confidence": round(conf, 2)})


@app.route("/predict_live")
def predict_live():

    def generate():
        cap = cv2.VideoCapture(0)
        prediction_buffer.clear()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))

            label, conf, bbox = predict_frame(frame, mode="video")
            now = time.time()

            # smoothing buffer
            if label not in ("NO_HAND", "LOW_CONF"):
                prediction_buffer.append(label)
            else:
                prediction_buffer.append("---")

            last   = prediction_buffer[-1] if prediction_buffer else "---"
            votes  = prediction_buffer.count(last)
            stable = (last != "---") and (votes >= SMOOTH_MIN_VOTES)

            # ── Sentence builder (thread-safe via state dict) ──
            with state_lock:
                if stable:
                    state["last_gesture_time"] = now

                    if last != state["last_added"]:
                        # new gesture detected — start hold timer
                        state["last_added"]      = last
                        state["last_added_time"] = now

                    elif (now - state["last_added_time"]) >= LETTER_HOLD_TIME:
                        # held long enough — add to sentence
                        sentence_buffer.append(last)
                        state["sentence"]        = "".join(sentence_buffer)
                        state["last_added_time"] = now + 999  # block re-add
                        speak_text(last)

                else:
                    if label == "NO_HAND":
                        state["last_added"] = ""
                        gap = now - state["last_gesture_time"]
                        if gap >= WORD_GAP_TIME:
                            if sentence_buffer and sentence_buffer[-1] != " ":
                                sentence_buffer.append(" ")
                                state["sentence"] = "".join(sentence_buffer)
                            state["last_gesture_time"] = now
                    else:
                        state["last_added"] = ""

                state["prediction"] = last if stable else (
                    "Not sure..." if label == "LOW_CONF" else
                    "Show your hand" if label == "NO_HAND" else
                    "Detecting " + label + "..."
                )
                state["confidence"] = conf

            # draw bounding box
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0,255,0) if stable else (0,200,255), 2)

            # hold progress bar
            if stable:
                with state_lock:
                    held = min(now - state["last_added_time"], LETTER_HOLD_TIME)
                if held < LETTER_HOLD_TIME:
                    bar = int((held / LETTER_HOLD_TIME) * 200)
                    cv2.rectangle(frame, (15,75), (215,95), (50,50,50), -1)
                    cv2.rectangle(frame, (15,75), (15+bar,95), (0,255,0), -1)
                    cv2.putText(frame, "Hold to add", (220,90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # overlay
            display = state["prediction"]
            color   = (0,255,0) if stable else (
                      (0,165,255) if label=="LOW_CONF" else
                      (0,0,255)   if label=="NO_HAND"  else
                      (0,200,255))

            cv2.rectangle(frame, (0,0), (640,70), (0,0,0), -1)
            cv2.putText(frame, display, (15,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)

            # sentence bar at bottom
            cv2.rectangle(frame, (0,440), (640,480), (0,0,0), -1)
            sent = state["sentence"][-40:] if state["sentence"] else ""
            cv2.putText(frame, "Sentence: " + sent, (10,468),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            _, buf = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/get_prediction")
def get_prediction():
    with state_lock:
        return jsonify(state.copy())


@app.route("/get_sentence")
def get_sentence():
    with state_lock:
        return jsonify({"sentence": state["sentence"]})


@app.route("/clear_sentence")
def clear_sentence():
    with state_lock:
        sentence_buffer.clear()
        state["sentence"]         = ""
        state["last_added"]       = ""
        state["last_added_time"]  = 0.0
    return jsonify({"status": "cleared"})


@app.route("/delete_last")
def delete_last():
    with state_lock:
        if sentence_buffer:
            sentence_buffer.pop()
        state["sentence"] = "".join(sentence_buffer)
    return jsonify({"sentence": state["sentence"]})


@app.route("/speak_sentence")
def speak_sentence():
    with state_lock:
        sentence = state["sentence"].strip()
    if sentence:
        speak_text(sentence)
    return jsonify({"status": "speaking", "sentence": sentence})


@app.route("/stop_webcam")
def stop_webcam():
    return jsonify({"status": "stopped"})


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)