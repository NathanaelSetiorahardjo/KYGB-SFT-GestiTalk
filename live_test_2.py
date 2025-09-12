"""
bisindo_app_with_key.py

Desktop app for live Bisindo recognition + Gemini interpretation with a modern UI.

Paste your Gemini API key directly into the GENAI_API_KEY variable below.

Controls:
- Start ★ : opens camera and starts recording gestures (auto-commits stable labels to sentence)
- Restart Translation : clears both the rolling sentence and the captured sentence (restart detection)
- Interpret with Gemini : send captured sentence to Gemini (if GENAI_API_KEY set) or fallback beautifier
- Confirm : accept interpreted sentence (saves to ./captures/confirmed_sentences.txt)
- Stop & Exit : stop camera and close the application

Requirements:
pip install opencv-python mediapipe tensorflow Pillow numpy customtkinter
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
from PIL import Image
from tkinter import messagebox
import customtkinter

# ---------- CONFIGURATION ----------
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

# ---------------------------
# !!! PENTING / IMPORTANT !!!
# Paste your Gemini API key here (between the quotes).
# Ganti tulisan "YOUR_GEMINI_API_KEY_HERE" dengan API Key Anda.
GENAI_API_KEY = "AIzaSyDaS8yX3GeVPY6kVhD8-13HF_pIS8raOfQ"
# ---------------------------


# ---------- GLOBAL INITIALIZATION & LOADING ----------

# 1. Gemini API Setup
USE_GENAI = False
genai_client = None
try:
    if GENAI_API_KEY and GENAI_API_KEY != "AIzaSyDaS8yX3GeVPY6kVhD8-13HF_pIS8raOfQ":
        import google.generativeai as genai
        genai.configure(api_key=GENAI_API_KEY)
        USE_GENAI = True
        genai_client = genai
        print("Gemini support enabled (google.generativeai).")
    else:
        print("GENAI_API_KEY not set — using local beautifier.")
except Exception as e:
    print(f"Gemini client unavailable or failed to import; using local beautifier. Err: {e}")

# 2. Model Loading
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

# 3. Label Loading
if os.path.exists(LABELS_JSON):
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    labels = DEFAULT_LABELS
    print(f"Warning: {LABELS_JSON} not found. Using defaults: {labels}")
num_classes = len(labels)

# 4. Mode Determination
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
    raise ValueError(f"Unsupported model input shape: {model_input_shape}")

# 5. MediaPipe Hands Initializer
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------- HELPER FUNCTIONS ----------

def normalize_hand(landmarks, w, h):
    if landmarks is None:
        return np.zeros(63, dtype=np.float32)
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
    center = pts[0].copy()
    pts[:, :2] -= center[:2]
    px = pts[:, 0] * w
    py = pts[:, 1] * h
    scale = max(px.max() - px.min(), py.max() - py.min(), 1e-3)
    pts[:, :2] /= (scale / max(w, h))
    return pts.flatten().astype(np.float32)

def build_feat_from_hands(res, w, h):
    left = None
    right = None
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
        xs, ys = [], []
        for lm_set in res.multi_hand_landmarks:
            for lm in lm_set.landmark:
                xs.append(lm.x)
                ys.append(lm.y)
        if xs and ys:
            x_min = int(max(0, min(xs) * w - 20))
            x_max = int(min(w, max(xs) * w + 20))
            y_min = int(max(0, min(ys) * h - 20))
            y_max = int(min(h, max(ys) * h + 20))
            if y_min < y_max and x_min < x_max:
                roi = frame[y_min:y_max, x_min:x_max]
    if H_target and W_target:
        roi_resized = cv2.resize(roi, (W_target, H_target))
    else:
        roi_resized = roi
    rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    return rgb.astype("float32") / 255.0

def simple_beautify(words):
    if not words:
        return ""
    s = " ".join(words).strip().lower()
    if s.startswith("apa ") or s == "apa":
        return s.capitalize() + "?"
    else:
        return s.capitalize() + "."

def interpret_with_gemini(words, prefer_lang="id"):
    if not words:
        return "", None
    joined = " ".join(words)

    if not USE_GENAI or genai_client is None:
        return simple_beautify(words), "Gemini not configured"

    prompt = (
        f"terjemahkan kata-kata bahasa isyarat BISINDO yang terpotong-potong ini menjadi kalimat {prefer_lang} yang baku dan natural. "
        f"Hanya tampilkan hasil kalimatnya saja. Input: '{joined}'"
    )

    try:
        # Preferred modern API call
        if hasattr(genai_client, "GenerativeModel"):
            model = genai_client.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            if hasattr(response, 'text') and response.text:
                return response.text.strip(), None
            else:
                return simple_beautify(words), f"Gemini response was empty or blocked. Parts: {response.parts}"
        else:
            return simple_beautify(words), "Unsupported google-generativeai library version."

    except Exception as e:
        return simple_beautify(words), f"GenAI error: {e}"


# ---------- MAIN APPLICATION CLASS ----------

class BisindoApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Bisindo Live — Recorder + Interpreter")
        self.geometry("800x720")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- Theme ---
        customtkinter.set_appearance_mode("Dark")
        customtkinter.set_default_color_theme("blue")

        # --- Layout Configuration ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Widgets ---
        self.video_label = customtkinter.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        control_frame = customtkinter.CTkFrame(self)
        control_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        control_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        font_button = ("Arial", 14, "bold")
        self.btn_start = customtkinter.CTkButton(control_frame, text="Start ★", command=self.start_camera, font=font_button, height=40)
        self.btn_start.grid(row=0, column=0, padx=6, pady=10, sticky="ew")

        self.btn_restart = customtkinter.CTkButton(control_frame, text="Restart Translation", command=self.restart_translation, font=font_button, height=40)
        self.btn_restart.grid(row=0, column=1, padx=6, pady=10, sticky="ew")

        self.btn_interpret = customtkinter.CTkButton(control_frame, text="Interpret with Gemini", command=self.interpret_action, font=font_button, height=40)
        self.btn_interpret.grid(row=0, column=2, padx=6, pady=10, sticky="ew")

        self.btn_confirm = customtkinter.CTkButton(control_frame, text="Confirm", command=self.confirm_action, fg_color="#28a745", hover_color="#218838", font=font_button, height=40)
        self.btn_confirm.grid(row=0, column=3, padx=6, pady=10, sticky="ew")

        status_frame = customtkinter.CTkFrame(self)
        status_frame.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        font_label = ("Arial", 16)
        self.label_sentence = customtkinter.CTkLabel(status_frame, text="Detected: ", anchor="w", font=font_label)
        self.label_sentence.grid(row=0, column=0, sticky="we", padx=10, pady=5)

        self.label_captured = customtkinter.CTkLabel(status_frame, text="Captured: ", anchor="w", font=font_label)
        self.label_captured.grid(row=1, column=0, sticky="we", padx=10, pady=5)

        self.label_interpreted = customtkinter.CTkLabel(status_frame, text="Interpreted: ", anchor="w", font=(font_label[0], font_label[1], "bold"), text_color="#52a2f2")
        self.label_interpreted.grid(row=2, column=0, sticky="we", padx=10, pady=(5, 10))

        self.btn_stop = customtkinter.CTkButton(self, text="Stop & Exit", command=self.on_close, fg_color="#d9534f", hover_color="#c9302c", height=35)
        self.btn_stop.grid(row=3, column=0, padx=10, pady=(5,10), sticky="ew")

        # --- Runtime State Variables ---
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_feats = deque(maxlen=SEQ_LEN)
        self.prob_buffer = deque(maxlen=PROB_SMOOTH_LEN)
        self.label_buffer = deque(maxlen=LABEL_STABLE_LEN)
        self.running_mean = {lbl: 0.5 for lbl in labels}
        self.sentence = []
        self.captured = []
        self.prev_gray = None
        self.motion_vals = deque(maxlen=MOTION_WINDOW)
        self.interpreted_sentence = ""
        self.lock = threading.Lock()

    def start_camera(self):
        if self.running:
            messagebox.showinfo("Info", "Camera already running")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        self.running = True
        with self.lock:
            self.frame_feats.clear(); self.prob_buffer.clear(); self.label_buffer.clear()
            self.sentence.clear(); self.captured.clear()
            self.interpreted_sentence = ""; self.prev_gray = None; self.motion_vals.clear()
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        self.after(30, self._update_frame)

    def stop_camera(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image=None)
        self._refresh_labels()

    def restart_translation(self):
        with self.lock:
            self.sentence.clear()
            self.captured.clear()
            self.interpreted_sentence = ""
            self.prob_buffer.clear()
            self.label_buffer.clear()
        self._refresh_labels()

    def interpret_action(self):
        with self.lock:
            captured_copy = list(self.captured)
        if not captured_copy:
            messagebox.showinfo("Interpret", "No captured words — perform gestures first.")
            return
        self.label_interpreted.configure(text="Interpreting with Gemini...")
        self.update_idletasks()
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
        self.restart_translation()

    def on_close(self):
        self.stop_camera()
        self.destroy()

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
            
            try:
                self._process_frame_for_prediction(frame)
            except Exception as e:
                print(f"Processing error: {e}")
            time.sleep(0.01)

    def _process_frame_for_prediction(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        motion = float(np.mean(np.abs(gray - self.prev_gray))) if self.prev_gray is not None else 0.0
        self.prev_gray = gray
        self.motion_vals.append(motion)
        mean_motion = float(np.mean(self.motion_vals))

        rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb_small)

        presence_count = 0
        if MODE == "landmarks":
            feat, presence_count = build_feat_from_hands(res, w, h)
            self.frame_feats.append(feat)
        else: # FRAMES mode
            img = preprocess_frame_roi_for_model(frame, res, H_target=H, W_target=W)
            self.frame_feats.append(img)
            if res.multi_hand_landmarks: presence_count = len(res.multi_hand_landmarks)

        if len(self.frame_feats) == SEQ_LEN:
            if mean_motion >= MOTION_THRESHOLD or presence_count > 0:
                X_in = np.expand_dims(np.stack(self.frame_feats, axis=0), axis=0)
                probs = model.predict(X_in, verbose=0)[0]
                self.prob_buffer.append(probs)
                avg_probs = np.mean(self.prob_buffer, axis=0)
                top_idx = int(np.argmax(avg_probs))
                top_conf = float(avg_probs[top_idx])
                top_label = labels[top_idx]

                prev = self.running_mean[top_label]
                self.running_mean[top_label] = (1 - ALPHA_RUNNING) * prev + ALPHA_RUNNING * top_conf
                dynamic_thr = max(self.running_mean[top_label] + BASE_MARGIN, CONF_MIN)

                if top_conf >= dynamic_thr:
                    self.label_buffer.append(top_label)
                    if len(self.label_buffer) == self.label_buffer.maxlen and all(x == self.label_buffer[0] for x in self.label_buffer):
                        with self.lock:
                            if not self.sentence or self.sentence[-1] != self.label_buffer[0]:
                                self.sentence.append(self.label_buffer[0])
                                self.captured.append(self.label_buffer[0])
                        self.label_buffer.clear()
                else:
                    self.label_buffer.clear()
            else:
                self.prob_buffer.clear(); self.label_buffer.clear()

    def _update_frame(self):
        if not self.running:
            return
        try:
            frame = self.frame_queue.get_nowait()
            if frame is not None:
                display = frame.copy()
                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb_small)
                if res.multi_hand_landmarks:
                    for hand_lms in res.multi_hand_landmarks:
                        mp_draw.draw_landmarks(display, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                ctk_img = customtkinter.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.video_label.configure(image=ctk_img)
        except queue.Empty:
            pass # No new frame, just wait for the next cycle
        except Exception as e:
            print(f"UI update error: {e}")
        
        self._refresh_labels()
        self.after(30, self._update_frame)

    def _refresh_labels(self):
        with self.lock:
            detected_text = " ".join(self.sentence[-20:])
            captured_text = " ".join(self.captured[-20:])
            interp = self.interpreted_sentence
        
        self.label_sentence.configure(text=f"Detected: {detected_text}")
        self.label_captured.configure(text=f"Captured: {captured_text}")
        self.label_interpreted.configure(text=f"Interpreted: {interp}")


# ---------- MAIN EXECUTION BLOCK ----------
if __name__ == "__main__":
    app = BisindoApp()
    app.mainloop()
