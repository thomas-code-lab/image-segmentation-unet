import os
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .capture import FrameGrabber
from .model import UNet

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.getenv(
    "FRONTEND_DIST",
    os.path.abspath(os.path.join(APP_ROOT, "..", "frontend", "dist")),
)
OBS_CAMERA_INDEX = int(os.getenv("OBS_CAMERA_INDEX", "1"))
USE_FULL_FRAME = os.getenv("USE_FULL_FRAME", "1") == "1"
ROI_X = int(os.getenv("ROI_X", "3"))
ROI_Y = int(os.getenv("ROI_Y", "4"))
ROI_W = int(os.getenv("ROI_W", "507"))
ROI_H = int(os.getenv("ROI_H", "504"))
OVERLAY_ALPHA = float(os.getenv("OVERLAY_ALPHA", "0.35"))
DEFAULT_KERAS_MODEL_PATH = os.path.abspath(
    os.path.join(APP_ROOT, "..", "models", "nerve_segmentation.keras")
)
KERAS_MODEL_PATH = os.getenv("KERAS_MODEL_PATH", "/model/nerve_segmentation.keras")
if not os.path.exists(KERAS_MODEL_PATH) and os.path.exists(DEFAULT_KERAS_MODEL_PATH):
    KERAS_MODEL_PATH = DEFAULT_KERAS_MODEL_PATH

KERAS_THRESHOLD = float(os.getenv("KERAS_THRESHOLD", "0.8"))
RUN_EVERY_N_FRAMES = int(os.getenv("RUN_EVERY_N_FRAMES", "1"))

POST_MORPH = os.getenv("POST_MORPH", "1") == "1"
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

GATE_ENABLED = os.getenv("GATE_ENABLED", "1") == "1"
GATE_CROP_MARGIN = float(os.getenv("GATE_CROP_MARGIN", "0.10"))
GATE_MEAN_MIN = float(os.getenv("GATE_MEAN_MIN", "12.0"))
GATE_STD_MIN = float(os.getenv("GATE_STD_MIN", "7.0"))

KEEP_LARGEST_COMPONENT = os.getenv("KEEP_LARGEST_COMPONENT", "1") == "1"
MIN_AREA_RATIO = float(os.getenv("MIN_AREA_RATIO", "0.006"))
MAX_AREA_RATIO = float(os.getenv("MAX_AREA_RATIO", "0.20"))

BORDER_REJECT = os.getenv("BORDER_REJECT", "1") == "1"
BORDER_MARGIN = int(os.getenv("BORDER_MARGIN", "6"))

MEAN_PROB_ENABLED = os.getenv("MEAN_PROB_ENABLED", "1") == "1"
MEAN_PROB_MIN = float(os.getenv("MEAN_PROB_MIN", "0.85"))

TEMPORAL_VOTE = os.getenv("TEMPORAL_VOTE", "1") == "1"
VOTE_K = int(os.getenv("VOTE_K", "3"))
VOTE_MIN = int(os.getenv("VOTE_MIN", "2"))

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
MODEL_PATH = os.getenv("MODEL_PATH", "")
INFER_SIZE = int(os.getenv("INFER_SIZE", "256"))
USE_CUDA = os.getenv("USE_CUDA", "1") == "1"

app = FastAPI()

grabber = FrameGrabber(camera_index=CAMERA_INDEX)
grabber.start()


def load_model() -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    model = UNet().to(device)
    model.eval()
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    return model


model = load_model()
keras_model = None
keras_input_h = None
keras_input_w = None
keras_area = None
if os.path.exists(KERAS_MODEL_PATH):
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
    keras_input_h = int(keras_model.input_shape[1])
    keras_input_w = int(keras_model.input_shape[2])
    keras_area = float(keras_input_h * keras_input_w)
    dummy = np.zeros((1, keras_input_h, keras_input_w, 1), dtype=np.float32)
    keras_model.predict(dummy, verbose=0)


def preprocess(frame: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor


def postprocess(mask: torch.Tensor, out_shape) -> np.ndarray:
    mask = torch.sigmoid(mask).squeeze(0).squeeze(0).cpu().numpy()
    mask = cv2.resize(mask, out_shape, interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255.0).clip(0, 255).astype(np.uint8)
    return mask


def preprocess_roi_to_model(gray_roi: np.ndarray) -> np.ndarray:
    img = cv2.resize(gray_roi, (keras_input_w, keras_input_h), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return x[None, ..., None]


def postprocess_mask(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 >= KERAS_THRESHOLD).astype(np.uint8) * 255
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
        y = out_img.shape[0] - 10
        for line in reversed(lines):
            (_, th), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(out_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y -= (th + baseline + 6)
    else:
        y = 25
        for line in lines:
            cv2.putText(out_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            (_, th), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y += (th + baseline + 6)


def mjpeg_stream():
    while True:
        frame = grabber.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )


@app.get("/video")
def video_feed() -> StreamingResponse:
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


def _open_obs_capture() -> cv2.VideoCapture:
    if os.name == "nt":
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(OBS_CAMERA_INDEX, backend)
            if cap.isOpened():
                return cap
        cap = cv2.VideoCapture(OBS_CAMERA_INDEX)
        if cap.isOpened():
            return cap
        # Fallback to default camera index if OBS index fails.
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        return cap
    if os.name == "posix" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(OBS_CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(OBS_CAMERA_INDEX)


def overlay_mjpeg_stream():
    cap = _open_obs_capture()
    if not cap.isOpened():
        raise RuntimeError("Cannot open OBS Virtual Camera.")

    fps_smooth = 0.0
    t_prev = time.time()
    frame_count = 0
    last_mask_roi = None
    infer_ms_smooth = 0.0
    mask_history = deque(maxlen=VOTE_K)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if USE_FULL_FRAME:
                roi = frame
                x0, y0 = 0, 0
            else:
                roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
                x0, y0 = ROI_X, ROI_Y

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

            did_infer = False
            infer_ms = None
            largest_area = 0
            area_ratio = 0.0
            mean_prob = 0.0
            vote_ready = (len(mask_history) == VOTE_K)

            if (not gated) and keras_model is not None and frame_count % RUN_EVERY_N_FRAMES == 0:
                t0 = time.time()
                x_in = preprocess_roi_to_model(gray)
                pred = keras_model.predict(x_in, verbose=0)
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

                if keras_area:
                    area_ratio = largest_area / keras_area

                if area_ratio < MIN_AREA_RATIO:
                    m256[:] = 0
                    largest_area = 0
                    bbox = None
                    area_ratio = 0.0

                if largest_area > 0 and area_ratio > MAX_AREA_RATIO:
                    m256[:] = 0
                    largest_area = 0
                    bbox = None
                    area_ratio = 0.0

                if BORDER_REJECT and bbox is not None and largest_area > 0:
                    bx, by, bw, bh = bbox
                    if (
                        bx < BORDER_MARGIN
                        or by < BORDER_MARGIN
                        or (bx + bw) > (keras_input_w - BORDER_MARGIN)
                        or (by + bh) > (keras_input_h - BORDER_MARGIN)
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
                    mask_history.append((m256 > 0).astype(np.uint8))
                    vote_ready = (len(mask_history) == VOTE_K)
                    if vote_ready:
                        vote = np.stack(mask_history, axis=0).sum(axis=0)
                        m256 = (vote >= VOTE_MIN).astype(np.uint8) * 255
                    else:
                        m256[:] = 0

                last_mask_roi = cv2.resize(
                    m256,
                    (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                t1 = time.time()
                infer_ms = (t1 - t0) * 1000.0
                infer_ms_smooth = (
                    infer_ms if infer_ms_smooth == 0 else (0.9 * infer_ms_smooth + 0.1 * infer_ms)
                )
                did_infer = True

            blended = overlay_mask_green(roi, last_mask_roi, OVERLAY_ALPHA)

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
            gate_txt = (
                f"gate(mean={gate_mean:.1f},std={gate_std:.1f})" if GATE_ENABLED else "gate(off)"
            )
            vote_txt = f"vote({len(mask_history)}/{VOTE_K})" if TEMPORAL_VOTE else "vote(off)"
            hud_lines = [
                (
                    "U-Net"
                    f" | FPS {fps_smooth:.1f}"
                    f" | infer {infer_ms_smooth:.1f}ms"
                    f" | {status}"
                    f" | N={RUN_EVERY_N_FRAMES}"
                    f" | thr={KERAS_THRESHOLD:.2f}"
                ),
                f"{gate_txt} | area={area_ratio*100:.2f}% | meanP={mean_prob:.2f} | {vote_txt}",
                (
                    f"ALPHA={OVERLAY_ALPHA:.2f}"
                    f" | minA={MIN_AREA_RATIO*100:.2f}%"
                    f" | maxA={MAX_AREA_RATIO*100:.0f}%"
                    f" | border={int(BORDER_REJECT)}"
                    f" | meanGate={int(MEAN_PROB_ENABLED)}"
                ),
            ]
            draw_hud(out, hud_lines, bottom_left=True)

            ok, jpg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )
            frame_count += 1
    finally:
        cap.release()


@app.get("/video/overlay")
def video_overlay_feed() -> StreamingResponse:
    return StreamingResponse(overlay_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/mask")
async def mask_socket(ws: WebSocket):
    await ws.accept()
    device = next(model.parameters()).device
    try:
        while True:
            frame = grabber.get_frame()
            if frame is None:
                await ws.send_text("nop")
                await ws.receive_text()
                continue
            tensor = preprocess(frame).to(device)
            with torch.no_grad():
                out = model(tensor)
            mask = postprocess(out, (frame.shape[1], frame.shape[0]))
            ok, png = cv2.imencode(".png", mask)
            if not ok:
                continue
            await ws.send_bytes(png.tobytes())
            await ws.receive_text()
    except Exception:
        await ws.close()


def _frontend_available() -> bool:
    return os.path.exists(os.path.join(FRONTEND_DIST, "index.html"))


if _frontend_available():
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def frontend_index():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

    @app.get("/{path:path}")
    def frontend_spa(path: str):
        # Let explicit API routes handle their own paths.
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))



