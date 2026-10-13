"""Minimal driving-game HUD drawn on top of generated frames."""
import cv2
import numpy as np

from .routes import speed_profile

FONT = cv2.FONT_HERSHEY_DUPLEX
YELLOW = (255, 204, 0)   # taxi yellow (RGB)
WHITE = (255, 255, 255)


def _panel(img, x0, y0, x1, y1, alpha=0.45):
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - alpha)).astype(np.uint8)


def _key(img, x, y, label, on, size=38):
    color = YELLOW if on else (90, 90, 90)
    cv2.rectangle(img, (x, y), (x + size, y + size), color, -1 if on else 2, cv2.LINE_AA)
    tc = (20, 20, 20) if on else (200, 200, 200)
    cv2.putText(img, label, (x + 11, y + 27), FONT, 0.8, tc, 1, cv2.LINE_AA)


def _wheel(img, cx, cy, r, angle_deg):
    cv2.circle(img, (cx, cy), r, WHITE, 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 6, WHITE, -1, cv2.LINE_AA)
    for base in (90, 210, 330):
        a = np.deg2rad(base + angle_deg)
        p = (int(cx + r * np.cos(a)), int(cy - r * np.sin(a)))
        cv2.line(img, (cx, cy), p, WHITE, 3, cv2.LINE_AA)


def draw_hud(frames, keyboard, mouse, label="NEW YORK CITY", top_speed=35, units="MPH"):
    """frames: (T,H,W,3) uint8 RGB. keyboard (T,6), mouse (T,2) numpy arrays."""
    T = len(frames)
    keyboard = np.asarray(keyboard)[:T]
    mouse = np.asarray(mouse)[:T]
    speeds = speed_profile(keyboard, top_speed)
    out = []
    steer = 0.0
    for i, f in enumerate(frames):
        img = np.ascontiguousarray(f.copy())
        H, W = img.shape[:2]
        k = keyboard[min(i, len(keyboard) - 1)]
        yaw = mouse[min(i, len(mouse) - 1)][1]
        steer += (np.clip(yaw / 0.1, -1, 1) * 90 - steer) * 0.25  # smoothed wheel angle

        # Location banner (top-left)
        _panel(img, 16, 16, 16 + 14 * len(label) + 40, 62)
        cv2.circle(img, (34, 39), 7, YELLOW, -1, cv2.LINE_AA)
        cv2.putText(img, label, (50, 48), FONT, 0.75, WHITE, 1, cv2.LINE_AA)

        # Title (top-right)
        title = "MATRIX NEW YORK"
        (tw, _), _ = cv2.getTextSize(title, FONT, 0.6, 1)
        cv2.putText(img, title, (W - tw - 20, 42), FONT, 0.6, YELLOW, 1, cv2.LINE_AA)

        # Keys (bottom-left)
        bx, by = 24, H - 104
        _panel(img, bx - 10, by - 10, bx + 140, H - 12)
        _key(img, bx + 46, by, "W", k[0] > 0.5)
        _key(img, bx + 2, by + 44, "A", k[2] > 0.5 or yaw < -0.02)
        _key(img, bx + 46, by + 44, "S", k[1] > 0.5)
        _key(img, bx + 90, by + 44, "D", k[3] > 0.5 or yaw > 0.02)

        # Speedometer + steering wheel (bottom-right)
        _panel(img, W - 250, H - 112, W - 14, H - 12)
        cv2.putText(img, f"{int(speeds[i]):3d}", (W - 240, H - 40), FONT, 1.6, WHITE, 2, cv2.LINE_AA)
        cv2.putText(img, units, (W - 236, H - 20), FONT, 0.5, YELLOW, 1, cv2.LINE_AA)
        _wheel(img, W - 64, H - 62, 36, -steer)
        out.append(img)
    return np.stack(out)
