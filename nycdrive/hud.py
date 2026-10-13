"""Driving HUD overlay: location, pedal/steering inputs, speed and steering-wheel angle."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
ACCENT = (255, 204, 0)   # RGB
WHITE = (255, 255, 255)
GREY = (110, 110, 110)


def _darken(img, x0, y0, x1, y1, alpha=0.45):
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1].astype(np.float32) * (1 - alpha)).astype(np.uint8)


def _key(img, x, y, label, on, s):
    cv2.rectangle(img, (x, y), (x + s, y + s), ACCENT if on else GREY, -1 if on else 1, cv2.LINE_AA)
    cv2.putText(img, label, (x + int(s * 0.3), y + int(s * 0.72)), FONT, s / 48, (20, 20, 20) if on else (200, 200, 200), 1, cv2.LINE_AA)


def _wheel(img, cx, cy, r, angle_deg):
    cv2.circle(img, (cx, cy), r, WHITE, max(1, r // 12), cv2.LINE_AA)
    for base in (0, 180, 270):
        a = np.deg2rad(base - angle_deg)
        cv2.line(img, (cx, cy), (int(cx + r * np.cos(a)), int(cy + r * np.sin(a))), WHITE, max(1, r // 12), cv2.LINE_AA)


def draw_hud(frames, controls, speed_mps, label, units="mph"):
    """frames (T,H,W,3) uint8; controls (N,2) [throttle, steer]; speed_mps (N,) or None to omit the readout."""
    controls = np.asarray(controls)
    out = []
    for i, f in enumerate(frames):
        img = np.ascontiguousarray(f.copy())
        H, W = img.shape[:2]
        s = max(18, H // 20)                     # key size scales with resolution
        j = min(i, len(controls) - 1)
        thr, st = controls[j]

        sc = H / 704
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6 * sc, 1)
        _darken(img, 10, 10, 26 + tw, 22 + th)
        cv2.putText(img, label, (18, 16 + th), FONT, 0.6 * sc, WHITE, 1, cv2.LINE_AA)

        bx, by = 14, H - 2 * s - 22
        _darken(img, bx - 6, by - 6, bx + 3 * s + 18, H - 10)
        _key(img, bx + s + 6, by, "W", thr > 0.05, s)
        _key(img, bx, by + s + 6, "A", st < -0.05, s)
        _key(img, bx + s + 6, by + s + 6, "S", thr < -0.05, s)
        _key(img, bx + 2 * s + 12, by + s + 6, "D", st > 0.05, s)

        pw = int((190 if speed_mps is not None else 90) * sc)
        _darken(img, W - pw - 10, H - int(84 * sc), W - 10, H - 10)
        if speed_mps is not None:
            spd = speed_mps[j] * (2.23694 if units == "mph" else 3.6)
            cv2.putText(img, f"{spd:3.0f}", (W - pw, H - int(34 * sc)), FONT, 1.3 * sc, WHITE, 2, cv2.LINE_AA)
            cv2.putText(img, units, (W - pw + 4, H - int(16 * sc)), FONT, 0.45 * sc, ACCENT, 1, cv2.LINE_AA)
        r = int(28 * sc)
        _wheel(img, W - 14 - r - int(10 * sc), H - 10 - r - int(14 * sc), r, st * 120)
        out.append(img)
    return np.stack(out)
