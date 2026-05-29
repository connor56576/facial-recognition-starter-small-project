import cv2
import numpy as np
import mediapipe as mp
from .base import BaseEffectGroup

mp_hands = mp.solutions.hands

#hyperparameters
PINCH_THRESHOLD = 15
PINCH_COOLDOWN = 0.6
BOX_TIMEOUT = 200
LERP_FACTOR = 1.0
BOX_PAD = 30
CORNER_LEN = 20
CORNER_COLOUR = (0, 220, 180)

class HandEffectGroup(BaseEffectGroup):
    """
    All logic from main.py has been moved here.
    This group provides effects that are triggered by pinching with either hand, and applies them to the area between the wrists.
    """
    def __init__(self):
        # Map names to their corresponding method
        self.effects_map = {
            "none": self._none,
            "glitch": self._glitch,
            "chromatic": self._chromatic,
            "sketch": self._sketch,
            "thermal": self._thermal,
            "cartoon": self._cartoon,
            "invert": self._invert,
            "pixelated": self._pixelated,
            "edge detect": self._edge_detect,
        }
        self._effect_list = list(self.effects_map.keys())
        self._pinch_state = {0: False, 1: False} #suprisingly useful for how simple it is
        self._pinch_cooldown = 0.0
        self._last_box = None
        self._missing_frames = 0
        self._gesture_data = {"hands": []} 
   
    @property
    def effect_list(self):
        return self._effect_list


    def process_landmarks(self, results, frame_shape): 
        """Convert mediapipe results into a dict of hand landmark positions."""
        h, w = frame_shape[:2]
        hands = []

        if results.multi_hand_landmarks: # 
            for lm in results.multi_hand_landmarks:
                hands.append({
                    "wrist": self._px(lm, mp_hands.HandLandmark.WRIST, w, h), 
                    "index": self._px(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP, w, h), 
                    "thumb": self._px(lm, mp_hands.HandLandmark.THUMB_TIP, w, h),
                })

        self._gesture_data = {"hands": hands}
        return self._gesture_data

    def update(self, gesture_data, now):
        """return true if pinch """
        advance = False

        for i, hand in enumerate(gesture_data["hands"][:2]):
            pinched = self._dist(hand["index"], hand["thumb"]) < PINCH_THRESHOLD #not hardcoded anymore :)
            if pinched and not self._pinch_state.get(i, False) and now > self._pinch_cooldown: #cooldown
                advance= True
                self._pinch_cooldown = now + PINCH_COOLDOWN
            self._pinch_state[i] = pinched # update regardless
        return advance

    def process_frame(self, frame, effect_name):
        """
        applies everything
        returns frame
        """
        display = frame.copy()
        rect = self._compute_rect(frame.shape)

        if rect is not None:
            x1, y1, x2, y2 = rect
            roi = frame[y1:y2, x1:x2].copy()

            if roi.size > 0:
                display[y1:y2, x1:x2] = self.effects_map.get(effect_name, self._none)(roi)

            # rect border
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # corner accents
            #annoying but im keeping it
            
            for cx, cy, dx, dy in [
                (x1, y1,  1,  1),
                (x2, y1, -1,  1),
                (x1, y2,  1, -1),
                (x2, y2, -1, -1), #
            ]:
                cv2.line(display, (cx, cy), (cx + dx * CORNER_LEN, cy),CORNER_COLOUR, 3) #took too long
                cv2.line(display, (cx, cy), (cx,cy + dy * CORNER_LEN),CORNER_COLOUR, 3) # yes 

        return display

    

    #smoothness stuff                                             
   

    def _compute_rect(self, frame_shape):
        h, w = frame_shape[:2]
        wrists = [hand["wrist"] for hand in self._gesture_data["hands"]]

        if len(wrists) == 2:
            self._missing_frames = 0
            x1 = max(0, min(wrists[0][0], wrists[1][0]) - BOX_PAD) # left
            y1 = max(0, min(wrists[0][1], wrists[1][1]) - BOX_PAD) #  top
            x2 = min(w, max(wrists[0][0], wrists[1][0]) + BOX_PAD) # right
            y2 = min(h, max(wrists[0][1], wrists[1][1]) + BOX_PAD) # bottom

            if self._last_box is None:
                self._last_box = (x1, y1, x2, y2)
            else:
                self._last_box = self._lerp(self._last_box, (x1, y1, x2, y2))

        else:
            self._missing_frames += 1
            if self._missing_frames >= BOX_TIMEOUT: #200
                self._last_box = None

        return self._last_box
    #interpolation strat
    def _lerp(self, old, new):
        return tuple(int(o + (n - o) * LERP_FACTOR) for o, n in zip(old, new)) #could just remove this. 
        #fancy code for just returning a fractional distance, not sure if needed yet 
                                                        

    @staticmethod
    def _px(hand_lm, landmark_id, w, h): #normalise to pixel
        lm = hand_lm.landmark[landmark_id]
        return int(lm.x * w), int(lm.y * h)

    @staticmethod
    def _dist(p1, p2): #distance, obviously
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


    #EFFECTS 
    #alot of these effects are me just messing around with opencv functions to see what looks cool

    def _none(self, roi):
        return roi

    def _glitch(self, roi):
        out = roi.copy()
        h = roi.shape[0]
        for _ in range(6):
            y = np.random.randint(0, max(1, h - 4))
            thickness = np.random.randint(2, 6)
            shift = np.random.randint(-20, 20)
            strip = out[y : y + thickness].copy()
            out[y : y + thickness] = np.roll(strip, shift, axis=1)
        # Color channel swap
        out[:, :, 0], out[:, :, 2] = out[:, :, 2].copy(), out[:, :, 0].copy()
        return out

    def _chromatic(self, roi):
        b, g, r = cv2.split(roi)
        shift = 6
        M_r = np.float32([[1, 0, shift], [0, 1, 0]])
        M_b = np.float32([[1, 0, -shift], [0, 1, 0]])



        h2, w2 = roi.shape[:2]
        r = cv2.warpAffine(r, M_r, (w2, h2))
        b = cv2.warpAffine(b, M_b, (w2, h2))
        return cv2.merge([b, g, r])

    def _sketch(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def _thermal(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        norm = cv2.equalizeHist(gray)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def _cartoon(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(roi, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    def _invert(self, roi):
        return cv2.bitwise_not(roi)

    def _edge_detect(self, roi): #SO COOL
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def _pixelated(self, roi): #new favourite effect

        h, w = roi.shape[:2] #dimensions of hand area

        #controls pixel intensity
        scale = max(4, int(min(w, h) / 25)) #the smaller the hand area the less pixelated 

        # Downscaled size
        small_w=max(1, w // scale)
        small_h=max(1, h // scale)

        #shrink image
        small = cv2.resize( roi,(small_w, small_h),
            interpolation=cv2.INTER_LINEAR
        )

        #scale back up
        pixelated = cv2.resize(small, (w, h),
            interpolation=cv2.INTER_NEAREST
        )


        return pixelated