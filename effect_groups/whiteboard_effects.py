import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from .base import BaseEffectGroup

mp_hands = mp.solutions.hands
colours = {
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255), # not really white, used for erasing
}

class WhiteboardEffects(BaseEffectGroup):
    """
    Creates a virtual whiteboard that you can draw on by pinching and moving your hand. Pinching toggles between draw and erase mode.
    """
    def __init__(self, width=1280, height=720):
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255 # start with blank white canvas
        self.prev_point = None
        self.brush_size = 2
        self.eraser_size = 40
        self._pinch_state = False
        self._pinch_cooldown = 0.0

        #all useful for reducing jitter

        #rolling average over last 5 positions for smoothing
        self._smooth_n=5 
        self._tip_history = deque(maxlen=self._smooth_n) # stores recent positions for smoothing

        self._pen_up_frames = 0
        self._pen_up_threshold = 3  #must see pen up for 3 consecutive frames

        #keep prev point briefly after losing detection
        #similar to boundingbox but stronger
        self._lost_frames = 0
        self._lost_threshold = 4

    @property
    def effect_list(self):
        return ["draw", "erase"]

    def _smooth_point(self, raw_px):
        """returns weighted average of recent positions"""



        self._tip_history.append(raw_px)


        if len(self._tip_history) == 1: # initial
            return raw_px
        

        #more recent frames count more
        weights = np.array([i + 1 for i in range(len(self._tip_history))], dtype=float) # weights from 1 to n, so more recent frames have higher weight
        #not really necessary

        weights /= weights.sum() #normalisng
        points = np.array(self._tip_history, dtype=float)

        avg = (points * weights[:, None]).sum(axis=0)
        return (int(avg[0]), int(avg[1]))
    



    def process_landmarks(self, results, frame_shape):
        
        if not results.multi_hand_landmarks:
            return {} 

        h, w = frame_shape[:2]
        lm = results.multi_hand_landmarks[0]
        #no loop for just one hand
         # very similar to other effect logic
        tip = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] #pen
        thumb = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
        middle = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP] 
        middle_mcp = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        tip_px = (int(tip.x * w),int(tip.y * h)) #covnersion
        thumb_px = (int(thumb.x * w), int(thumb.y * h))

        #distance relative to hand size so it works anywhere in frame
        index_mcp = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP] #knuckle
        hand_scale = np.hypot( 
            (tip.x - index_mcp.x) * w, 
            (tip.y - index_mcp.y) * h
        )
        pinch_dist = np.hypot(tip_px[0] - thumb_px[0], tip_px[1] - thumb_px[1]) 
        pinched = pinch_dist < hand_scale * 0.35   #test more

        pen_up = middle.y < middle_mcp.y  

        return {
            "tip": tip_px,
            "pinched": pinched,
            "pen_up": pen_up,
        }

    def update(self, gesture_data, now, effect_name):
        #
        if not gesture_data:
            self._lost_frames += 1
            #only fully reset after 4 consecutive missed frames
            if self._lost_frames >= self._lost_threshold:



                self.prev_point = None
                self._tip_history.clear()


            return False

        self._lost_frames = 0  #reset counter

        #pinching toggles draw or erase mode
        pinched = gesture_data.get("pinched", False)
        if pinched and not self._pinch_state and (now > self._pinch_cooldown):
            self._pinch_cooldown = now + 0.6 #same as filters
            self.prev_point = None
            self._pinch_state = pinched
            return True

        # pen up detection logic
        raw_pen_up = gesture_data.get("pen_up", False)
        if raw_pen_up:
            self._pen_up_frames += 1
        else:
            self._pen_up_frames = 0
        pen_up = self._pen_up_frames >= self._pen_up_threshold # pen up if 3 or more frames

        if pinched or pen_up:
            self.prev_point = None
            return False

        raw_tip = gesture_data["tip"]
        tip = self._smooth_point(raw_tip) 



        if effect_name == "draw":
            color = colours["BLACK"]  #only black for now
        else:
            color = colours["WHITE"]



        if effect_name == "draw":
            size  = self.brush_size
        else:
            size = self.eraser_size

        if self.prev_point is not None:
            #interpolation 
            dist = np.hypot(tip[0] - self.prev_point[0], tip[1] - self.prev_point[1]) 
            if dist < 30:  #dont teleport instead draw line, change to lower
                cv2.line(self.canvas, self.prev_point, tip, color, size * 2, 
                         lineType=cv2.LINE_AA)
        else: 
            cv2.circle(self.canvas, tip, size, color, -1) #start with dot if no previous point

        self.prev_point = tip
        self._pinch_state = pinched ###
        if pinched:
            return True
        else: 
            return False

    def process_frame(self, frame, effect_name): # 
        #similar to other copy of function but majority of logic is in update instead, need to change in future
        h, w = frame.shape[:2] 

        if self.canvas.shape[:2] != (h, w):
            self.canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        display = frame.copy()
        ink_mask = cv2.inRange(self.canvas, (0, 0, 0), (50, 50, 50)) #mask of black pixels. needs good lighting
        display[ink_mask > 0] = self.canvas[ink_mask > 0] 
        return display

    def clear(self):
        self.canvas[:] = 255#clears everything
        self._tip_history.clear()
        self.prev_point = None