import os
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))
DEFAULT_MODEL = os.path.join(REPO_ROOT, "models", "nerve_segmentation.keras")

# ========= CONFIG (env-first) =========
IDX = int(os.getenv("OBS_CAMERA_INDEX", "1"))
USE_FULL_FRAME = os.getenv("USE_FULL_FRAME", "1") == "1"
X = int(os.getenv("ROI_X", "3"))
Y = int(os.getenv("ROI_Y", "4"))
W = int(os.getenv("ROI_W", "507"))
H = int(os.getenv("ROI_H", "504"))

MODEL_PATH = os.getenv("KERAS_MODEL_PATH", "/model/nerve_segmentation.keras")
if not os.path.exists(MODEL_PATH) and os.path.exists(DEFAULT_MODEL):
    MODEL_PATH = DEFAULT_MODEL

MODEL_IMG_SIZE = int(os.getenv("KERAS_INPUT_SIZE", "256"))
ALPHA = float(os.getenv("OVERLAY_ALPHA", "0.35"))
THRESH = float(os.getenv("KERAS_THRESHOLD", "0.85"))
RUN_EVERY_N_FRAMES = int(os.getenv("RUN_EVERY_N_FRAMES", "2"))

POST_MORPH = os.getenv("POST_MORPH", "1") == "1"
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

GATE_ENABLED = os.getenv("GATE_ENABLED", "1") == "1"
GATE_CROP_MARGIN = float(os.getenv("GATE_CROP_MARGIN", "0.10"))
GATE_MEAN_MIN = float(os.getenv("GATE_MEAN_MIN", "12.0"))
GATE_STD_MIN = float(os.getenv("GATE_STD_MIN", "7.0"))

KEEP_LARGEST_COMPONENT = os.getenv("KEEP_LARGEST_COMPONENT", "1") == "1"
MIN_AREA_RATIO_256 = float(os.getenv("MIN_AREA_RATIO", "0.006"))
MAX_AREA_RATIO_256 = float(os.getenv("MAX_AREA_RATIO", "0.20"))

BORDER_REJECT = os.getenv("BORDER_REJECT", "1") == "1"
BORDER_MARGIN_256 = int(os.getenv("BORDER_MARGIN", "6"))

MEAN_PROB_ENABLED = os.getenv("MEAN_PROB_ENABLED", "1") == "1"
MEAN_PROB_MIN = float(os.getenv("MEAN_PROB_MIN", "0.92"))

TEMPORAL_VOTE = os.getenv("TEMPORAL_VOTE", "1") == "1"
VOTE_K = int(os.getenv("VOTE_K", "3"))
VOTE_MIN = int(os.getenv("VOTE_MIN", "2"))
MASK_HISTORY = deque(maxlen=VOTE_K)

HUD_BOTTOM = True
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.55
HUD_THICK = 2
HUD_MARGIN_X = 10
HUD_MARGIN_Y = 10
HUD_LINE_GAP = 6

# ========= LOAD MODEL =========
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Loaded model:", MODEL_PATH)
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

model_input_h = MODEL_IMG_SIZE
model_input_w = MODEL_IMG_SIZE
model_input_shape = model.input_shape
if isinstance(model_input_shape, (list, tuple)) and model_input_shape:
    model_input_shape = model_input_shape[0]
if isinstance(model_input_shape, tuple) and len(model_input_shape) >= 3:
    model_input_h = int(model_input_shape[1] or MODEL_IMG_SIZE)
    model_input_w = int(model_input_shape[2] or MODEL_IMG_SIZE)


def run_model(x_in: np.ndarray) -> np.ndarray:
    x_tensor = tf.convert_to_tensor(x_in, dtype=tf.float32)
    y = model([x_tensor], training=False)
    if isinstance(y, (list, tuple)):
        y = y[0]
    return y.numpy()


_dummy = np.zeros((1, model_input_h, model_input_w, 1), dtype=np.float32)
_ = run_model(_dummy)


def _open_capture(index: int) -> cv2.VideoCapture:
    if os.name == "nt":
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(index)
    if os.name == "posix" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(index)


def select_roi(frame: np.ndarray):
    if USE_FULL_FRAME:
        return frame, 0, 0
    h, w = frame.shape[:2]
    x1 = max(0, X)
    y1 = max(0, Y)
    x2 = min(w, X + W)
    y2 = min(h, Y + H)
    if x2 <= x1 or y2 <= y1:
        return frame, 0, 0
    return frame[y1:y2, x1:x2], x1, y1


def preprocess_roi_to_model(gray_roi: np.ndarray) -> np.ndarray:
    img = cv2.resize(gray_roi, (model_input_w, model_input_h), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return x[None, ..., None]


def postprocess_mask(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 >= THRESH).astype(np.uint8) * 255
    if POST_MORPH:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)
    return m


def overlay_mask_green(roi_bgr: np.ndarray, mask255: np.ndarray, alpha: float) -> np.ndarray:
    overlay = roi_bgr.copy()
    overlay[mask255 > 0] = (0, 255, 0)
    return cv2.addWeighted(overlay, alpha, roi_bgr, 1 - alpha, 0)


def central_crop_stats(gray_roi: np.ndarray, margin: float):
    h, w = gray_roi.shape[:2]
    dy = int(h * margin)
    dx = int(w * margin)
    y1, y2 = dy, max(dy + 1, h - dy)
    x1, x2 = dx, max(dx + 1, w - dx)
    crop = gray_roi[y1:y2, x1:x2]
    return float(crop.mean()), float(crop.std())


def gate_contact(gray_roi: np.ndarray):
    mean_v, std_v = central_crop_stats(gray_roi, GATE_CROP_MARGIN)
    ok = (mean_v >= GATE_MEAN_MIN) and (std_v >= GATE_STD_MIN)
    return ok, mean_v, std_v


def keep_largest_component(mask255: np.ndarray):
    bin01 = (mask255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask255), 0, None

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + int(np.argmax(areas))
    max_area = int(stats[max_label, cv2.CC_STAT_AREA])

    x = int(stats[max_label, cv2.CC_STAT_LEFT])
    y = int(stats[max_label, cv2.CC_STAT_TOP])
    w = int(stats[max_label, cv2.CC_STAT_WIDTH])
    h = int(stats[max_label, cv2.CC_STAT_HEIGHT])

    out = np.zeros_like(mask255)
    out[labels == max_label] = 255
    return out, max_area, (x, y, w, h)


def ensure_mask_shape(mask: np.ndarray, roi_shape_hw):
    h, w = roi_shape_hw
    if mask is None or mask.shape[:2] != (h, w):
        return np.zeros((h, w), dtype=np.uint8)
    return mask


def draw_hud(out_img: np.ndarray, lines, bottom_left=True):
    if not lines:
        return
    if bottom_left:
        y = out_img.shape[0] - HUD_MARGIN_Y
        for line in reversed(lines):
            (_, th), baseline = cv2.getTextSize(line, HUD_FONT, HUD_SCALE, HUD_THICK)
            cv2.putText(out_img, line, (HUD_MARGIN_X, y), HUD_FONT, HUD_SCALE, (255, 255, 255), HUD_THICK)
            y -= (th + baseline + HUD_LINE_GAP)
    else:
        y = 25
        for line in lines:
            cv2.putText(out_img, line, (HUD_MARGIN_X, y), HUD_FONT, HUD_SCALE, (255, 255, 255), HUD_THICK)
            (_, th), baseline = cv2.getTextSize(line, HUD_FONT, HUD_SCALE, HUD_THICK)
            y += (th + baseline + HUD_LINE_GAP)


def reset_temporal():
    MASK_HISTORY.clear()


cap = _open_capture(IDX)
if not cap.isOpened():
    raise RuntimeError("Cannot open OBS Virtual Camera. Make sure it is started.")

fps_smooth = 0.0
t_prev = time.time()
infer_ms_smooth = 0.0
frame_i = 0
last_mask_roi = None

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("No frame.")
        break

    roi, x0, y0 = select_roi(frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    last_mask_roi = ensure_mask_shape(last_mask_roi, (roi.shape[0], roi.shape[1]))

    gated = False
    gate_mean = 0.0
    gate_std = 0.0
    contact_ok = True

    if GATE_ENABLED:
        contact_ok, gate_mean, gate_std = gate_contact(gray)
        if not contact_ok:
            last_mask_roi.fill(0)
            gated = True

    frame_i += 1
    did_infer = False
    largest_area = 0
    area_ratio = 0.0
    mean_prob = 0.0
    vote_ready = (len(MASK_HISTORY) == VOTE_K)

    if (not gated) and (frame_i % RUN_EVERY_N_FRAMES == 0):
        t0 = time.time()
        x_in = preprocess_roi_to_model(gray)
        pred = run_model(x_in)
        pred = np.squeeze(pred)

        if pred.ndim == 3:
            pred = pred[..., 0]
        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected model output shape after squeeze: {pred.shape}")

        m256 = postprocess_mask(pred)

        bbox = None
        if KEEP_LARGEST_COMPONENT:
            m256, largest_area, bbox = keep_largest_component(m256)
        else:
            largest_area = int((m256 > 0).sum())

        area_ratio = largest_area / float(MODEL_IMG_SIZE * MODEL_IMG_SIZE)

        if area_ratio < MIN_AREA_RATIO_256:
            m256[:] = 0
            largest_area = 0
            bbox = None
            area_ratio = 0.0

        if largest_area > 0 and area_ratio > MAX_AREA_RATIO_256:
            m256[:] = 0
            largest_area = 0
            bbox = None
            area_ratio = 0.0

        if BORDER_REJECT and bbox is not None and largest_area > 0:
            bx, by, bw, bh = bbox
            if (
                bx < BORDER_MARGIN_256
                or by < BORDER_MARGIN_256
                or (bx + bw) > (MODEL_IMG_SIZE - BORDER_MARGIN_256)
                or (by + bh) > (MODEL_IMG_SIZE - BORDER_MARGIN_256)
            ):
                m256[:] = 0
                largest_area = 0
                bbox = None
                area_ratio = 0.0

        if MEAN_PROB_ENABLED and largest_area > 0:
            mean_prob = float(pred[m256 > 0].mean())
            if mean_prob < MEAN_PROB_MIN:
                m256[:] = 0
                largest_area = 0
                bbox = None
                area_ratio = 0.0
                mean_prob = 0.0

        if TEMPORAL_VOTE:
            MASK_HISTORY.append((m256 > 0).astype(np.uint8))
            vote_ready = (len(MASK_HISTORY) == VOTE_K)
            if vote_ready:
                vote = np.stack(MASK_HISTORY, axis=0).sum(axis=0)
                m256 = (vote >= VOTE_MIN).astype(np.uint8) * 255
            else:
                m256[:] = 0

        last_mask_roi = cv2.resize(m256, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

        t1 = time.time()
        infer_ms = (t1 - t0) * 1000.0
        infer_ms_smooth = infer_ms if infer_ms_smooth == 0 else (0.9 * infer_ms_smooth + 0.1 * infer_ms)
        did_infer = True

    blended = overlay_mask_green(roi, last_mask_roi, ALPHA)

    if USE_FULL_FRAME:
        out = blended
    else:
        out = frame.copy()
        out[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]] = blended

    t_now = time.time()
    dt = t_now - t_prev
    t_prev = t_now
    if dt > 0:
        inst = 1.0 / dt
        fps_smooth = inst if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * inst)

    status = "GATE" if gated else ("INFER" if did_infer else "HOLD")
    gate_txt = f"gate(mean={gate_mean:.1f},std={gate_std:.1f})" if GATE_ENABLED else "gate(off)"
    vote_txt = f"vote({len(MASK_HISTORY)}/{VOTE_K})" if TEMPORAL_VOTE else "vote(off)"

    hud_lines = [
        f"UNET | FPS {fps_smooth:.1f} | infer {infer_ms_smooth:.1f}ms | {status} | N={RUN_EVERY_N_FRAMES} | thr={THRESH:.2f}",
        f"{gate_txt} | area={area_ratio*100:.2f}% | meanP={mean_prob:.2f} | {vote_txt}",
        f"ALPHA={ALPHA:.2f} | minA={MIN_AREA_RATIO_256*100:.2f}% | maxA={MAX_AREA_RATIO_256*100:.0f}% | border={int(BORDER_REJECT)} | meanGate={int(MEAN_PROB_ENABLED)}",
    ]
    draw_hud(out, hud_lines, bottom_left=HUD_BOTTOM)

    cv2.imshow("Live Overlay (U-Net)", out)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

    if key == ord("f"):
        USE_FULL_FRAME = not USE_FULL_FRAME
        last_mask_roi = None
        reset_temporal()

    if key == ord("]"):
        RUN_EVERY_N_FRAMES = min(10, RUN_EVERY_N_FRAMES + 1)
    if key == ord("["):
        RUN_EVERY_N_FRAMES = max(1, RUN_EVERY_N_FRAMES - 1)

    if key == ord("=") or key == ord("+"):
        THRESH = min(0.95, THRESH + 0.02)
        last_mask_roi = None
        reset_temporal()
    if key == ord("-") or key == ord("_"):
        THRESH = max(0.05, THRESH - 0.02)
        last_mask_roi = None
        reset_temporal()

    if key == ord("g"):
        GATE_ENABLED = not GATE_ENABLED
        last_mask_roi = None
        reset_temporal()

    if key == ord("v"):
        TEMPORAL_VOTE = not TEMPORAL_VOTE
        last_mask_roi = None
        reset_temporal()

    if key == ord("b"):
        BORDER_REJECT = not BORDER_REJECT
        last_mask_roi = None
        reset_temporal()

    if key == ord("p"):
        MEAN_PROB_ENABLED = not MEAN_PROB_ENABLED
        last_mask_roi = None
        reset_temporal()

    if key == ord("u"):
        GATE_MEAN_MIN = min(80.0, GATE_MEAN_MIN + 1.0)
        last_mask_roi = None
        reset_temporal()
    if key == ord("j"):
        GATE_MEAN_MIN = max(0.0, GATE_MEAN_MIN - 1.0)
        last_mask_roi = None
        reset_temporal()

    if key == ord("i"):
        GATE_STD_MIN = min(50.0, GATE_STD_MIN + 0.5)
        last_mask_roi = None
        reset_temporal()
    if key == ord("k"):
        GATE_STD_MIN = max(0.0, GATE_STD_MIN - 0.5)
        last_mask_roi = None
        reset_temporal()

    if key == ord("o"):
        MIN_AREA_RATIO_256 = min(0.05, MIN_AREA_RATIO_256 + 0.0005)
        last_mask_roi = None
        reset_temporal()
    if key == ord("l"):
        MIN_AREA_RATIO_256 = max(0.0, MIN_AREA_RATIO_256 - 0.0005)
        last_mask_roi = None
        reset_temporal()

    if key == ord("9"):
        MEAN_PROB_MIN = min(0.99, MEAN_PROB_MIN + 0.01)
        last_mask_roi = None
        reset_temporal()
    if key == ord("8"):
        MEAN_PROB_MIN = max(0.50, MEAN_PROB_MIN - 0.01)
        last_mask_roi = None
        reset_temporal()

cap.release()
cv2.destroyAllWindows()
