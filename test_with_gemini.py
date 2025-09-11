# live_test_gemini.py
import os
import json
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
import google.generativeai as genai

# ====== CONFIG (tweak if needed) ======
MODEL_CANDIDATES = [
    "bisindo_landmarks.keras"
]
LABELS_JSON = "cache_landmarks/labels.json"  # try this first
DEFAULT_LABELS = ["Halo", "Kamu", "Apa", "Dimana", "Duduk"]  # fallback
SEQ_LEN_OVERRIDE = None
PROB_SMOOTH_LEN = 8
LABEL_STABLE_LEN = 4
ALPHA_RUNNING = 0.12
BASE_MARGIN = 0.09
MOTION_WINDOW = 6
MOTION_THRESHOLD = 0.008
CONF_MIN = 0.35
SHOW_TOPK = True

# ====== auto-find model ======
model_path = None
for p in MODEL_CANDIDATES:
    if os.path.exists(p):
        model_path = p
        break
if model_path is None:
    raise FileNotFoundError("No model found. Put a .keras or .h5 model file in the working folder.")

print("Loading model:", model_path)
model = tf.keras.models.load_model(model_path)
model_input_shape = model.input_shape
print("Model input shape:", model_input_shape)

# ====== load labels ======
if os.path.exists(LABELS_JSON):
    labels = json.load(open(LABELS_JSON, "r", encoding="utf-8"))
else:
    labels = DEFAULT_LABELS
    print("Warning: labels.json not found. Using default labels:", labels)

num_classes = len(labels)

# ====== determine mode (landmarks vs frames) ======
mode = None
if len(model_input_shape) == 3:
    mode = "landmarks"
    SEQ_LEN = SEQ_LEN_OVERRIDE or int(model_input_shape[1])
    feat_dim = int(model_input_shape[2])
    print(f"Operating in LANDMARKS mode: seq_len={SEQ_LEN}, feat_dim={feat_dim}")
elif len(model_input_shape) == 5:
    mode = "frames"
    SEQ_LEN = SEQ_LEN_OVERRIDE or int(model_input_shape[1])
    H = int(model_input_shape[2]); W = int(model_input_shape[3]); C = int(model_input_shape[4])
    print(f"Operating in FRAMES mode: seq_len={SEQ_LEN}, frame_size=({H},{W},{C})")
else:
    raise ValueError(f"Unsupported model input shape: {model_input_shape}")

# ====== mediapipe setup ======
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, model_complexity=0,
                       max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ====== runtime buffers & state ======
frame_feats = deque(maxlen=SEQ_LEN)
prob_buffer = deque(maxlen=PROB_SMOOTH_LEN)
label_buffer = deque(maxlen=LABEL_STABLE_LEN)
running_mean = {lbl: 0.5 for lbl in labels}
sentence = []
prev_gray = None
motion_vals = deque(maxlen=MOTION_WINDOW)

# ====== Gemini setup ======
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # <-- put your API key here
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

def unjumble_with_gemini(words_list, lang="id"):
    words = " ".join(words_list)
    if lang == "id":
        prompt = f"Susun kata-kata ini menjadi kalimat bahasa Indonesia yang benar: {words}"
    else:
        prompt = f"Rearrange these words into a grammatically correct sentence: {words}"
    response = model_gemini.generate_content(prompt)
    return response.text.strip()

# ====== open webcam ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found")

print("Live test started — Press SPACE to toggle detection, 'q' to quit.")

collecting = False
display_text = ""   # what will be shown on screen (raw words or Gemini sentence)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # motion calc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    if prev_gray is not None:
        motion = float(np.mean(np.abs(gray - prev_gray)))
    else:
        motion = 0.0
    prev_gray = gray
    motion_vals.append(motion)
    mean_motion = float(np.mean(motion_vals))

    # mediapipe detection
    rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb_small)

    # build features
    presence_count = 0
    if mode == "landmarks":
        feat, presence_count = build_feat_from_hands(res, w, h)
        frame_feats.append(feat)
    else:
        img = preprocess_frame_roi_for_model(frame, res, H_target=H, W_target=W)
        frame_feats.append(img)

    # draw landmarks
    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    final_label = "NoGesture"
    final_conf = 0.0

    if len(frame_feats) == SEQ_LEN:
        if mean_motion >= MOTION_THRESHOLD or presence_count > 0:
            if mode == "landmarks":
                X_in = np.expand_dims(np.stack(frame_feats, axis=0), axis=0)
            else:
                X_in = np.expand_dims(np.stack(frame_feats, axis=0), axis=0)

            probs = model.predict(X_in, verbose=0)[0]
            prob_buffer.append(probs)
            avg_probs = np.mean(prob_buffer, axis=0)

            top_idx = int(np.argmax(avg_probs))
            top_conf = float(avg_probs[top_idx])
            top_label = labels[top_idx]

            prev = running_mean[top_label]
            running_mean[top_label] = (1 - ALPHA_RUNNING) * prev + ALPHA_RUNNING * top_conf
            dynamic_thr = running_mean[top_label] + BASE_MARGIN
            dynamic_thr = max(dynamic_thr, CONF_MIN)

            if top_conf >= dynamic_thr:
                label_buffer.append(top_label)
                if len(label_buffer) == label_buffer.maxlen and all(x == label_buffer[0] for x in label_buffer):
                    if not sentence or sentence[-1] != label_buffer[0]:
                        if collecting:  # only collect when ON
                            sentence.append(label_buffer[0])
                            display_text = " ".join(sentence[-8:])
                    label_buffer.clear()
                final_label = top_label
                final_conf = top_conf
            else:
                final_label = "NoGesture"
                final_conf = top_conf
        else:
            prob_buffer.clear()
            label_buffer.clear()
            final_label = "NoGesture"
            final_conf = 0.0

    # overlays
    cv2.putText(frame, f"Pred: {final_label} ({final_conf:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, "Sentence: " + display_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, f"Motion:{mean_motion:.4f} Hands:{presence_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

    cv2.imshow("LIVE BISINDO TEST", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE toggles
        collecting = not collecting
        if collecting:
            print("\n🟢 Started collecting new sentence...")
            sentence = []
            display_text = ""
        else:
            print("🔴 Stopped collecting. Sending to Gemini...")
            if sentence:
                result = unjumble_with_gemini(sentence, lang="id")
                display_text = result  # replace raw words
                print("✨ Gemini Sentence:", result)
            else:
                print("No words collected.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
