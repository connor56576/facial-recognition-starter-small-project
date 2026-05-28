import cv2
import mediapipe as mp
import numpy as np
import time


# Mediapipe setup 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# Effects
EFFECTS = [
    "none",
    "glitch",
    "chromatic",
    "sketch",
    "thermal",
    "cartoon",
    "invert",]

effect_index = 0

def apply_effect(roi, name):
    # roi = region of interest 
    if name == "none":
        return roi
    #custom effects. Can really do whatever you want
    #alot of these effects are me just messing around with opencv functions to see what looks cool
    elif name == "glitch": 
        out = roi.copy()
        h = roi.shape[0] 
        for _ in range(6):
            y = np.random.randint(0, max(1, h - 4)) 
            thickness = np.random.randint(2, 6)
            shift = np.random.randint(-20, 20)
            strip = out[y : y + thickness].copy()
            out[y : y + thickness] = np.roll(strip, shift, axis=1)
        #colour channel swap
        out[:, :, 0], out[:, :, 2] = out[:, :, 2].copy(), out[:, :, 0].copy()
        return out

    elif name == "chromatic":
        b, g, r = cv2.split(roi)
        shift = 6
        M_r = np.float32([[1, 0, shift], [0, 1, 0]]) 
        M_b = np.float32([[1, 0, -shift], [0, 1, 0]])


        h2, w2 = roi.shape[:2] #
        r = cv2.warpAffine(r, M_r, (w2, h2)) #
        b = cv2.warpAffine(b, M_b, (w2, h2))
        return cv2.merge([b, g, r])

    elif name == "sketch":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256) 
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    #this ones my favourite
    elif name == "thermal":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        norm = cv2.equalizeHist(gray)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    
    elif name == "cartoon":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(roi, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)
    
    elif name == "invert":
        return cv2.bitwise_not(roi)

    return roi

    #removed halftone effect as a nested for loop was too slow
    
# Gesture helpers 

def wrist_pos(hand_landmarks, w, h):
    lm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST] 
    return int(lm.x * w), int(lm.y * h) # normalise

def index_tip(hand_landmarks, w, h):
    lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] # built in functions 
    return int(lm.x * w), int(lm.y * h)

def thumb_tip(hand_landmarks, w, h):
    lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    return int(lm.x * w), int(lm.y * h)

def dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) 
# State
pinch_state = {0: False, 1: False} #per hand
pinch_cooldown = 0.0


PINCH_THRESHOLD = 40 # px – index and thumb this close = pinch

# Helpers
def next_effect():
    global effect_index 
    effect_index = (effect_index + 1) % len(EFFECTS)

def overlay_text(frame, text, pos, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2) # should really have more fonts
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# Main loop
video = cv2.VideoCapture(0)
cv2.namedWindow("Hand Canvas", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = time.time()

while True:
    success, frame = video.read()
    if not success:
        break

    frame = cv2.flip(frame, 1) # mirror for natural feel
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    now = time.time()
    fps = (1.0 / max(now - prev_time, 1e-6)) #avoids div by zero
    prev_time = now
 
    rect = None # the rectangle this frame
    wrists = []
    hand_data = [] # list of (wrist_xy, index_xy, thumb_xy)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            wx, wy = wrist_pos(lm, w, h) 
            ix, iy = index_tip(lm, w, h)
            tx, ty = thumb_tip(lm, w, h)
            wrists.append((wx, wy))
            hand_data.append(((wx, wy), (ix, iy), (tx, ty)))

        # Rectangle from two hand
        if len(wrists) == 2:
            x1 = min(wrists[0][0], wrists[1][0])
            y1 = min(wrists[0][1], wrists[1][1])
            x2 = max(wrists[0][0], wrists[1][0])
            y2 = max(wrists[0][1], wrists[1][1])
            # pad a bit so the rectangle feels roomy
            pad = 30
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            if (x2 - x1) > 40 and (y2 - y1) > 40:
                rect = (x1, y1, x2, y2)
                
        #pinch detection
        for i, (_, (ix, iy), (tx, ty)) in enumerate(hand_data[:2]): #change variable names to be more readable
            pinched_now = dist((ix, iy), (tx, ty)) < PINCH_THRESHOLD
            if pinched_now and not pinch_state.get(i, False) and now > pinch_cooldown:
                next_effect()
                pinch_cooldown = now + 0.6 # change to not hard coded
            pinch_state[i] = pinched_now
    #apply effect inside rectangl
    display = frame.copy()

    if rect is not None:
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2].copy()
        if roi.size > 0:
            processed = apply_effect(roi, EFFECTS[effect_index])
            display[y1:y2, x1:x2] = processed

        # draw rectangle border
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

        #corner accents
        corner_len = 20
        col = (0, 220, 180) # make everythong more readable again in future
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(display, (cx, cy), (cx + dx * corner_len, cy), col, 3)
            cv2.line(display, (cx, cy), (cx, cy + dy * corner_len), col, 3)

    #hud 
    #dont hard code this in future
    effect_name = EFFECTS[effect_index].upper()
    overlay_text(display, f"Effect: {effect_name}", (12, 30), scale=0.7, color=(0, 220, 180))
    overlay_text(display, f"[{effect_index + 1}/{len(EFFECTS)}]", (12, 58), scale=0.5, color=(180, 180, 180))
    overlay_text(display, f"FPS: {fps:.1f}", (w - 120, 30), scale=0.7, color=(0, 220, 180))
    overlay_text(display, "SPACE: cycle | Q: quit", (12, h - 14), scale=0.45, color=(160, 160, 160))
   
    cv2.imshow("Hand Canvas", display)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q") or cv2.getWindowProperty("Hand Canvas", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord(" "): #space to skip
        next_effect()

video.release()
cv2.destroyAllWindows()
