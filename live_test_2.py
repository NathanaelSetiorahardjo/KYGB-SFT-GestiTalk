"""
bisindo_app_with_key.py

Desktop app for live Bisindo recognition + Gemini interpretation.

Paste your Gemini API key directly into the GENAI_API_KEY variable below.

Controls:
- Start ★ : opens camera and starts recording gestures (auto-commits stable labels to sentence)
- Restart Translation : clears both the rolling sentence and the captured sentence (restart detection)
- Interpret : send captured sentence to Gemini (if GENAI_API_KEY set) or fallback beautifier
- Confirm : accept interpreted sentence (saves to ./captures/confirmed_sentences.txt)
- Stop : stop camera and close

Requirements:
pip install opencv-python mediapipe tensorflow Pillow numpy
Optional (Gemini): pip install google-generativeai

Model: put your .keras/.h5 model in the working directory (default name checked: bisindo_landmarks.keras)
Labels: optionally place cache_landmarks/labels.json
"""

import os
import json
import time
import threading
import queue
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ---------- CONFIG ----------
MODEL_CANDIDATES = ["bisindo_landmarks.keras"]
LABELS_JSON = "cache_landmarks/labels.json"
DEFAULT_LABELS = ["Halo", "Kamu", "Apa", "Dimana", "Duduk"]
SEQ_LEN_OVERRIDE = None
PROB_SMOOTH_LEN = 8
LABEL_STABLE_LEN = 4
ALPHA_RUNNING = 0.12
BASE_MARGIN = 0.09
MOTION_WINDOW = 6
MOTION_THRESHOLD = 0.008
CONF_MIN = 0.35
SHOW_TOPK = True

# ---------------------------
# Paste your Gemini API key here (between the quotes). Example:
# GENAI_API_KEY = "AIza..."  or "sk-..." depending on your provider/key format.
# If you leave it empty (""), the app will use a local fallback beautifier.
GENAI_API_KEY = "AIzaSyClPF7vC04cfnNaVu6mimY-OzPWi2DFHvw"  # <-- PUT YOUR GEMINI API KEY HERE
# ---------------------------

# ---------- MODEL LOAD ----------
model_path = None
for p in MODEL_CANDIDATES:
    if os.path.exists(p):
        model_path = p
        break
if model_path is None:
    raise FileNotFoundError("No model found. Put a .keras or .h5 model in the working folder (e.g. bisindo_landmarks.keras)")

print("Loading model:", model_path)
model = tf.keras.models.load_model(model_path)
model_input_shape = model.input_shape
print("Model input shape:", model_input_shape)

# load labels
if os.path.exists(LABELS_JSON):
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    labels = DEFAULT_LABELS
    print("Warning: labels.json not found. Using defaults:", labels)
num_classes = len(labels)

# determine mode
if len(model_input_shape) == 3:
    MODE = "landmarks"
    SEQ_LEN = SEQ_LEN_OVERRIDE or int(model_input_shape[1])
    FEAT_DIM = int(model_input_shape[2])
    print(f"Operating in LANDMARKS mode: seq_len={SEQ_LEN}, feat_dim={FEAT_DIM}")
elif len(model_input_shape) == 5:
    MODE = "frames"
    SEQ_LEN = SEQ_LEN_OVERRIDE or int(model_input_shape[1])
    H = int(model_input_shape[2]); W = int(model_input_shape[3]); C = int(model_input_shape[4])
    print(f"Operating in FRAMES mode: seq_len={SEQ_LEN}, frame_size=({H},{W},{C})")
else:
    raise ValueError("Unsupported model input shape: " + str(model_input_shape))

# mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, model_complexity=0,
                       max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------- HELPERS & STATE ----------
def normalize_hand(landmarks, w, h):
    if landmarks is None:
        return np.zeros(63, dtype=np.float32)
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
    center = pts[0].copy()
    pts[:, :2] -= center[:2]
    px = pts[:,0] * w; py = pts[:,1] * h
    scale = max(px.max()-px.min(), py.max()-py.min(), 1e-3)
    pts[:, :2] /= (scale / max(w, h))
    return pts.flatten().astype(np.float32)


def build_feat_from_hands(res, w, h):
    left = None; right = None
    if res.multi_hand_landmarks and res.multi_handedness:
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            if getattr(handed.classification[0], "label", "").lower() == "left":
                left = lm.landmark
            else:
                right = lm.landmark
    L = normalize_hand(left, w, h)
    R = normalize_hand(right, w, h)
    pres = np.array([1.0 if left is not None else 0.0, 1.0 if right is not None else 0.0], dtype=np.float32)
    return np.concatenate([L, R, pres]), int(pres.sum())


def preprocess_frame_roi_for_model(frame, res=None, H_target=None, W_target=None):
    h, w = frame.shape[:2]
    roi = frame
    if res and res.multi_hand_landmarks:
        xs=[]; ys=[]
        for lm_set in res.multi_hand_landmarks:
            for lm in lm_set.landmark:
                xs.append(lm.x); ys.append(lm.y)
        if xs and ys:
            x_min = int(max(0, min(xs) * w - 20))
            x_max = int(min(w, max(xs) * w + 20))
            y_min = int(max(0, min(ys) * h - 20))
            y_max = int(min(h, max(ys) * h + 20))
            try:
                roi = frame[y_min:y_max, x_min:x_max]
            except Exception:
                roi = frame
    if H_target and W_target:
        try:
            roi_resized = cv2.resize(roi, (W_target, H_target))
        except Exception:
            roi_resized = cv2.resize(roi, (W_target, H_target))
    else:
        roi_resized = roi
    rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    return (rgb.astype("float32") / 255.0)

# genai wrapper (best-effort)
USE_GENAI = False
genai_client = None

def simple_beautify(words):
    if not words:
        return ""
    s = " ".join(words).strip().lower()
    if s.startswith("apa " ) or s == "apa":
        return s.capitalize() + "?"
    else:
        return s.capitalize() + "."

try:
    if GENAI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GENAI_API_KEY)
        USE_GENAI = True
        genai_client = genai
        print("Gemini support enabled (google.generativeai).")
    else:
        print("GENAI_API_KEY not set — using local beautifier.")
except Exception as e:
    print("Gemini client unavailable or failed to import; using local beautifier. Err:", e)

def interpret_with_gemini(words, prefer_lang="id"):
    """
    Robust wrapper for Gemini (google.generativeai). Uses the modern API call
    and keeps older ones as fallbacks.
    Returns: (text, error_or_None)
    """
    if not words:
        return "", None
    joined = " ".join(words)

    if not USE_GENAI or genai_client is None:
        return simple_beautify(words), None

    prompt = (
        f"terjemahkan kata-kata yang kurang jelas ini menjadi kalimat yang baku {prefer_lang}  "
        f"Buat kalimat yang lengkap dan jelas dan terdengar lebih natural tampilkan kalimatnya yang sudah di proses saja. Input: '{joined}'"
    )

    try:
        # 1) PREFERRED MODERN API CALL (for google-generativeai v0.3.0 and newer)
        if hasattr(genai_client, "GenerativeModel"):
            # Use a fast and efficient model for this task
            model = genai_client.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            # The .text attribute is the simplest way to get the string output
            if hasattr(response, 'text') and response.text:
                return response.text.strip(), None
            else: # Handle cases where the response might be blocked or empty
                return simple_beautify(words), f"Gemini response was empty or blocked. Parts: {response.parts}"

        # --- The rest of the function remains as a fallback for very old library versions ---

        # 2) Older: responses API
        if hasattr(genai_client, "responses") and hasattr(genai_client.responses, "create"):
            resp = genai_client.responses.create(model="gemini-1.5", input=prompt)
            output_text = ""
            if hasattr(resp, "output_text") and resp.output_text:
                output_text = resp.output_text
            else:
                out = getattr(resp, "output", None)
                if out:
                    for item in out:
                        for c in getattr(item, "content", []):
                            if isinstance(c, dict) and "text" in c:
                                output_text += c["text"]
                            else:
                                output_text += getattr(c, "text", "") or ""
            output_text = (output_text or "").strip()
            if output_text:
                return output_text, None
            else:
                return simple_beautify(words), "Gemini (responses) returned empty"

        # 3) Older: generate_text
        if hasattr(genai_client, "generate_text"):
            resp = genai_client.generate_text(model="gemini-1.5", prompt=prompt, max_output_tokens=256)
            text_out = getattr(resp, "text", None) or str(resp)
            return (text_out.strip(), None) if text_out else (simple_beautify(words), "generate_text empty")

        # No usable API entry found after all checks
        return simple_beautify(words), "GenAI client available but no supported call found"

    except Exception as e:
        # Always return a fallback instead of crashing the app; report error for debugging
        return simple_beautify(words), f"GenAI error: {e}"
# ---------- APP CLASS ----------
class BisindoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bisindo Live — Recorder + Interpreter")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # UI elements
        self.video_label = tk.Label(self)
        self.video_label.grid(row=0, column=0, columnspan=4)

        self.btn_start = tk.Button(self, text="Start ★", width=12, command=self.start_camera)
        self.btn_start.grid(row=1, column=0, padx=6, pady=6)

        self.btn_restart = tk.Button(self, text="Restart Translation", width=18, command=self.restart_translation)
        self.btn_restart.grid(row=1, column=1, padx=6)

        self.btn_interpret = tk.Button(self, text="Interpret", width=12, command=self.interpret_action)
        self.btn_interpret.grid(row=1, column=2, padx=6)

        self.btn_confirm = tk.Button(self, text="Confirm", width=10, command=self.confirm_action)
        self.btn_confirm.grid(row=1, column=3, padx=6)

        self.label_sentence = tk.Label(self, text="Detected: ", anchor="w")
        self.label_sentence.grid(row=2, column=0, columnspan=4, sticky="we", padx=6)

        self.label_captured = tk.Label(self, text="Captured: ", anchor="w")
        self.label_captured.grid(row=3, column=0, columnspan=4, sticky="we", padx=6)

        self.label_interpreted = tk.Label(self, text="Interpreted: ", anchor="w", fg="blue")
        self.label_interpreted.grid(row=4, column=0, columnspan=4, sticky="we", padx=6, pady=(0,8))

        self.btn_stop = tk.Button(self, text="Stop", width=10, command=self.stop_camera)
        self.btn_stop.grid(row=5, column=3, sticky="e", padx=6, pady=6)

        # runtime variables
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)

        # model + runtime buffers
        self.frame_feats = deque(maxlen=SEQ_LEN)
        self.prob_buffer = deque(maxlen=PROB_SMOOTH_LEN)
        self.label_buffer = deque(maxlen=LABEL_STABLE_LEN)
        self.running_mean = {lbl: 0.5 for lbl in labels}
        self.sentence = []           # rolling committed sentence
        self.captured = []          # captured during this camera session (for interpret)
        self.prev_gray = None
        self.motion_vals = deque(maxlen=MOTION_WINDOW)
        self.interpreted_sentence = ""
        self.lock = threading.Lock()

    # UI actions
    def start_camera(self):
        if self.running:
            messagebox.showinfo("Info", "Camera already running")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        self.running = True
        # clear previous buffers
        with self.lock:
            self.frame_feats.clear(); self.prob_buffer.clear(); self.label_buffer.clear()
            self.sentence.clear(); self.captured.clear()
            self.interpreted_sentence = ""
            self.prev_gray = None; self.motion_vals.clear()
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        self.after(30, self._update_frame)

    def stop_camera(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.video_label.config(image="")
        self.label_sentence.config(text="Detected: ")
        self.label_captured.config(text="Captured: ")
        self.label_interpreted.config(text="Interpreted: ")

    def restart_translation(self):
        with self.lock:
            self.sentence.clear()
            self.captured.clear()
            self.interpreted_sentence = ""
            self.prob_buffer.clear(); self.label_buffer.clear()
        self._refresh_labels()

    def interpret_action(self):
        # blocking call: interpret current captured sequence
        with self.lock:
            captured_copy = list(self.captured)
        if not captured_copy:
            messagebox.showinfo("Interpret", "No captured words — perform gestures first.")
            return
        interpreted, err = interpret_with_gemini(captured_copy, prefer_lang="id")
        with self.lock:
            self.interpreted_sentence = interpreted
        self._refresh_labels()
        if err:
            messagebox.showwarning("Interpret note", f"Interpretation fallback/issue: {err}")

    def confirm_action(self):
        with self.lock:
            final_text = self.interpreted_sentence or " ".join(self.captured) or " ".join(self.sentence)
        if not final_text:
            messagebox.showinfo("Confirm", "Nothing to confirm.")
            return
        os.makedirs("captures", exist_ok=True)
        fname = "captures/confirmed_sentences.txt"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  ||  {final_text}\n")
        messagebox.showinfo("Confirmed", f"Saved to {fname}")

    def on_close(self):
        self.stop_camera()
        self.destroy()

    # background camera loop
    def _camera_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            # process frame for detection/prediction
            try:
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
                if self.prev_gray is not None:
                    motion = float(np.mean(np.abs(gray - self.prev_gray)))
                else:
                    motion = 0.0
                self.prev_gray = gray
                self.motion_vals.append(motion)
                mean_motion = float(np.mean(self.motion_vals))

                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb_small)

                presence_count = 0
                if MODE == "landmarks":
                    feat, presence_count = build_feat_from_hands(res, w, h)
                    self.frame_feats.append(feat)
                else:
                    img = preprocess_frame_roi_for_model(frame, res, H_target=H, W_target=W)
                    self.frame_feats.append(img)

                final_label = "NoGesture"; final_conf = 0.0
                if len(self.frame_feats) == SEQ_LEN:
                    if mean_motion >= MOTION_THRESHOLD or presence_count > 0:
                        if MODE == "landmarks":
                            X_in = np.expand_dims(np.stack(self.frame_feats, axis=0), axis=0)
                        else:
                            X_in = np.expand_dims(np.stack(self.frame_feats, axis=0), axis=0)
                        probs = model.predict(X_in, verbose=0)[0]
                        self.prob_buffer.append(probs)
                        avg_probs = np.mean(self.prob_buffer, axis=0)
                        top_idx = int(np.argmax(avg_probs))
                        top_conf = float(avg_probs[top_idx])
                        top_label = labels[top_idx]

                        prev = self.running_mean[top_label]
                        self.running_mean[top_label] = (1 - ALPHA_RUNNING) * prev + ALPHA_RUNNING * top_conf
                        dynamic_thr = self.running_mean[top_label] + BASE_MARGIN
                        dynamic_thr = max(dynamic_thr, CONF_MIN)

                        if top_conf >= dynamic_thr:
                            self.label_buffer.append(top_label)
                            if len(self.label_buffer) == self.label_buffer.maxlen and all(x == self.label_buffer[0] for x in self.label_buffer):
                                with self.lock:
                                    if not self.sentence or self.sentence[-1] != self.label_buffer[0]:
                                        self.sentence.append(self.label_buffer[0])
                                        self.captured.append(self.label_buffer[0])
                                self.label_buffer.clear()
                            final_label = top_label; final_conf = top_conf
                        else:
                            final_label = "NoGesture"; final_conf = top_conf
                    else:
                        self.prob_buffer.clear(); self.label_buffer.clear()
            except Exception as e:
                print("Processing error:", e)
            time.sleep(0.01)

    # UI frame update
    def _update_frame(self):
        try:
            frame = None
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            if frame is not None:
                display = frame.copy()
                try:
                    rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb_small)
                    if res.multi_hand_landmarks:
                        for hand_lms in res.multi_hand_landmarks:
                            mp_draw.draw_landmarks(display, hand_lms, mp_hands.HAND_CONNECTIONS)
                except Exception:
                    pass
                img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self._refresh_labels()
        except Exception as e:
            print("UI update error:", e)
        if self.running:
            self.after(30, self._update_frame)

    def _refresh_labels(self):
        with self.lock:
            detected_text = " ".join(self.sentence[-20:])
            captured_text = " ".join(self.captured[-20:])
            interp = self.interpreted_sentence
        self.label_sentence.config(text=f"Detected: {detected_text}")
        self.label_captured.config(text=f"Captured: {captured_text}")
        self.label_interpreted.config(text=f"Interpreted: {interp}")

# ---------- MAIN ----------
if __name__ == "__main__":
    app = BisindoApp()
    app.geometry("760x680")
    app.mainloop()
