# live_test.py
import os
import json
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter

# ====== CONFIG (tweak if needed) ======
MODEL_CANDIDATES = [
    "bisindo_landmarks.keras"
]
LABELS_JSON = "cache_landmarks/labels.json"  # try this first
DEFAULT_LABELS = ["Halo", "Kamu", "Apa", "Dimana", "Duduk"]  # fallback
SEQ_LEN_OVERRIDE = None    # set to int to force seq length, else uses model shape
PROB_SMOOTH_LEN = 8
LABEL_STABLE_LEN = 4
ALPHA_RUNNING = 0.12
BASE_MARGIN = 0.09
MOTION_WINDOW = 6
MOTION_THRESHOLD = 0.008
CONF_MIN = 0.35   # minimum allowed confidence to consider anything (very low because adaptive thresholds used)
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
# landmarks mode: model_input_shape like (None, T, F) or (None, T, 128)
# frames mode: (None, T, H, W, C)
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

# ====== mediapipe setup (for landmarks + presence) ======
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, model_complexity=0,
                       max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ====== runtime buffers & state ======
frame_feats = deque(maxlen=SEQ_LEN)   # stores landmarks-features (landmarks mode) or frames (frames mode)
prob_buffer = deque(maxlen=PROB_SMOOTH_LEN)
label_buffer = deque(maxlen=LABEL_STABLE_LEN)
running_mean = {lbl: 0.5 for lbl in labels}
sentence = []
prev_gray = None
motion_vals = deque(maxlen=MOTION_WINDOW)

# ====== helper functions ======
def normalize_hand(landmarks, w, h):
    """Return 63-d vector for 21 landmarks (x,y,z) normalized relative to wrist & scale.
       If landmarks is None -> zeros."""
    if landmarks is None:
        return np.zeros(63, dtype=np.float32)
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
    center = pts[0].copy()
    pts[:, :2] -= center[:2]
    px = pts[:,0] * w
    py = pts[:,1] * h
    scale = max(px.max()-px.min(), py.max()-py.min(), 1e-3)
    pts[:, :2] /= (scale / max(w,h))
    return pts.flatten().astype(np.float32)

def build_feat_from_hands(res, w, h):
    left = None; right = None
    if res.multi_hand_landmarks and res.multi_handedness:
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            if handed.classification[0].label.lower() == "left":
                left = lm.landmark
            else:
                right = lm.landmark
    L = normalize_hand(left, w, h)
    R = normalize_hand(right, w, h)
    pres = np.array([1.0 if left is not None else 0.0, 1.0 if right is not None else 0.0], dtype=np.float32)
    return np.concatenate([L, R, pres]), int(pres.sum())

def preprocess_frame_roi_for_model(frame, res=None, H_target=None, W_target=None):
    """Return an RGB float32 frame cropped to hand bbox if res given, else center crop/resized."""
    h, w = frame.shape[:2]
    if res and res.multi_hand_landmarks:
        # compute bbox covering all detected hands
        xs=[]; ys=[]
        for lm_set in res.multi_hand_landmarks:
            for lm in lm_set.landmark:
                xs.append(lm.x); ys.append(lm.y)
        if xs and ys:
            x_min = int(max(0, min(xs) * w - 20))
            x_max = int(min(w, max(xs) * w + 20))
            y_min = int(max(0, min(ys) * h - 20))
            y_max = int(min(h, max(ys) * h + 20))
            roi = frame[y_min:y_max, x_min:x_max]
        else:
            roi = frame
    else:
        roi = frame
    # resize to target H,W
    if H_target and W_target:
        try:
            roi_resized = cv2.resize(roi, (W_target, H_target))
        except:
            roi_resized = cv2.resize(roi, (W_target, H_target))
    else:
        roi_resized = roi
    # convert BGR->RGB and normalize
    rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    return (rgb.astype("float32") / 255.0)

# ====== open webcam ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found")

print("Live test started — warming up. Keep your hands in view. Press 'q' to exit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror for natural selfie view
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

    # mediapipe detection (used for landmarks and for presence)
    rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb_small)

    # build features based on mode
    presence_count = 0
    if mode == "landmarks":
        feat, presence_count = build_feat_from_hands(res, w, h)
        frame_feats.append(feat)  # feat shape = 128 if two hands (63+63+2)
    else:  # frames mode
        img = preprocess_frame_roi_for_model(frame, res, H_target=H, W_target=W)
        frame_feats.append(img)  # img shape = (H,W,C)

    # draw landmarks for user feedback
    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    final_label = "NoGesture"
    final_conf = 0.0

    # decide when to predict: require buffer full and some presence or motion
    if len(frame_feats) == SEQ_LEN:
        # gating: prefer some motion or at least one hand presence
        if mean_motion >= MOTION_THRESHOLD or presence_count > 0:
            # prepare model input
            if mode == "landmarks":
                X_in = np.expand_dims(np.stack(frame_feats, axis=0), axis=0)  # (1,T,F)
            else:
                X_in = np.expand_dims(np.stack(frame_feats, axis=0), axis=0)  # (1,T,H,W,C)

            probs = model.predict(X_in, verbose=0)[0]  # (num_classes,)
            prob_buffer.append(probs)
            avg_probs = np.mean(prob_buffer, axis=0)

            top_idx = int(np.argmax(avg_probs))
            top_conf = float(avg_probs[top_idx])
            top_label = labels[top_idx]

            # adaptive running mean for that class
            prev = running_mean[top_label]
            running_mean[top_label] = (1 - ALPHA_RUNNING) * prev + ALPHA_RUNNING * top_conf
            dynamic_thr = running_mean[top_label] + BASE_MARGIN
            dynamic_thr = max(dynamic_thr, CONF_MIN)  # floor

            # accept if above dynamic threshold
            if top_conf >= dynamic_thr:
                label_buffer.append(top_label)
                # stable label check
                if len(label_buffer) == label_buffer.maxlen and all(x == label_buffer[0] for x in label_buffer):
                    # commit to sentence (avoid repeating identical consecutive)
                    if not sentence or sentence[-1] != label_buffer[0]:
                        sentence.append(label_buffer[0])
                    label_buffer.clear()
                final_label = top_label
                final_conf = top_conf
            else:
                # not confident enough: treat as NoGesture
                final_label = "NoGesture"
                final_conf = top_conf
        else:
            # low motion & no hands => NoGesture; also clear buffers to avoid stale votes
            prob_buffer.clear()
            label_buffer.clear()
            final_label = "NoGesture"
            final_conf = 0.0

    # overlays
    # show top-k if available
    debug_line = ""
    if len(prob_buffer):
        avg = np.mean(prob_buffer, axis=0)
        topk = np.argsort(avg)[-3:][::-1]
        debug_line = " | ".join(f"{labels[i]}:{avg[i]:.2f}" for i in topk)
    cv2.putText(frame, f"Pred: {final_label} ({final_conf:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, "Sentence: " + " ".join(sentence[-8:]), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, f"Motion:{mean_motion:.4f} Hands:{presence_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
    if SHOW_TOPK and debug_line:
        cv2.putText(frame, debug_line, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    cv2.imshow("LIVE BISINDO TEST", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
