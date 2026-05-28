import cv2
import mediapipe as mp
import numpy as np
import time
from effects import HandEffects

fx = HandEffects()
EFFECTS = fx.effect_list




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
    "invert",
    "pixelated",]

effect_index = 0


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


PINCH_THRESHOLD = 15 # px – index and thumb this close = pinch

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

last_box = None

LERP_FACTOR = 0.2 # smoothness of movement

#keep visible if detection fails
missing_frames = 0
BOX_TIMEOUT = 200

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
            #if (x2 - x1) > 40 and (y2 - y1) > 40: 
                #first frame
            if last_box is None:
                last_box = (x1, y1, x2, y2)

            else:
                lx1, ly1, lx2, ly2 = last_box

                #smoothing
                x1 = int(lx1 + (x1 - lx1) * LERP_FACTOR) #lerp between last box and new box by lerp factor to make movement smoother
                y1 = int(ly1 + (y1 - ly1) * LERP_FACTOR)
                x2 = int(lx2 + (x2 - lx2) * LERP_FACTOR)
                y2 = int(ly2 + (y2 - ly2) * LERP_FACTOR)

                last_box = (x1, y1, x2, y2)

            rect = last_box
            missing_frames = 0
        elif len(wrists) == 1:
            # one hand visible 
            missing_frames += 1
            if last_box is not None and missing_frames < BOX_TIMEOUT:
                rect = last_box
       
                    
        #pinch detection
        for i, (_, (ix, iy), (tx, ty)) in enumerate(hand_data[:2]): #change variable names to be more readable
            pinched_now = dist((ix, iy), (tx, ty)) < PINCH_THRESHOLD
            if pinched_now and not pinch_state.get(i, False) and now > pinch_cooldown:
                next_effect()
                pinch_cooldown = now + 0.6 # change to not hard coded
            pinch_state[i] = pinched_now
    
    else:
        missing_frames += 1 

        #keep showing previous rect
        if last_box is not None and missing_frames < BOX_TIMEOUT:
            rect = last_box
    
    #apply effect inside rectangl
    display = frame.copy()

    if rect is not None:
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2].copy()
        if roi.size > 0:
            processed = fx.apply(roi, EFFECTS[effect_index])
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
    overlay_text(display, "PINCH: cycle", (12, h - 40), scale=0.45, color=(180, 180, 180))
    overlay_text(display, "SPACE: cycle | Q: quit", (12, h - 14), scale=0.45, color=(160, 160, 160))
    #debug
    # overlay_text(display, "missing frames: " + str(missing_frames), (w - 150, h - 14), scale=0.45, color=(160, 160, 160))
    cv2.imshow("Hand Canvas", display)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q") or cv2.getWindowProperty("Hand Canvas", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord(" "): #space to skip
        next_effect()

video.release()
cv2.destroyAllWindows()
