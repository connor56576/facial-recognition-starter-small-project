import cv2
import mediapipe as mp
import time
from effect_groups.hand_effects import HandEffectGroup

# Mediapipe setup 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

class EffectManager: 
    """
    Manages multiple effect groups and their effects. Keeps track of the current group and effect, and provides methods to switch between them.
    The main loop interacts with this manager to get the current effect and apply it, without needing to know any details about how the groups work.

    """
    def __init__(self, groups: dict):
        self.groups = groups
        self.group_names = list(groups.keys())

        self.group_index = 0
        self.effect_index = 0

    def current_group_name(self):
        return self.group_names[self.group_index]
    
    def current_group(self):
        return self.groups[self.current_group_name()]

    def current_effect(self):
        group = self.groups[self.current_group_name()]
        return group.effect_list[self.effect_index]

    def next_effect(self):
        group = self.groups[self.current_group_name()]
        self.effect_index = (self.effect_index + 1) % len(group.effect_list)

    def next_group(self):
        self.group_index = (self.group_index + 1) % len(self.group_names)
        self.effect_index = 0  # reset on switch




def overlay_text(frame, text, pos, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2) # should really have more fonts
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

fx = EffectManager({"hand": HandEffectGroup()}) #instance of effectmanager

# Main loop
video = cv2.VideoCapture(0)
cv2.namedWindow("Effects", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Effects", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
 
    group = fx.current_group()

    gesture_data = group.process_landmarks(results, frame.shape)
    if group.update(gesture_data, now):
        fx.next_effect()

    display = group.process_frame(frame, fx.current_effect())
    
    #dont hard code this in future
    effect_name = fx.current_effect().upper()
    overlay_text(display, f"Group: {fx.current_group_name().upper()}", (12, 30), scale=0.7, color=(0, 220, 180))
    overlay_text(display, f"[{fx.group_index + 1}/{len(fx.group_names)}]", (12, 58), scale=0.5, color=(180, 180, 180))
    overlay_text(display, f"Effect: {effect_name}", (12, 86), scale=0.7, color=(0, 220, 180))
    overlay_text(display, f"[{fx.effect_index + 1}/{len(group.effect_list)}]", (12, 114), scale=0.5, color=(180, 180, 180))
    overlay_text(display, f"FPS: {fps:.1f}", (w - 120, 30), scale=0.7, color=(0, 220, 180))
    overlay_text(display, "PINCH: cycle", (12, h - 40), scale=0.45, color=(180, 180, 180))
    overlay_text(display, "SPACE: cycle | Q: quit", (12, h - 14), scale=0.45, color=(160, 160, 160))
    #debug
    # overlay_text(display, "missing frames: " + str(missing_frames), (w - 150, h - 14), scale=0.45, color=(160, 160, 160))
    cv2.imshow("Effects", display)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q") or cv2.getWindowProperty("Effects", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord(" "): #space to skip
        fx.next_effect()

video.release()
cv2.destroyAllWindows()
