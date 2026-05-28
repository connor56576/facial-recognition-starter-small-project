import cv2
import numpy as np

class HandEffects:
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
        }
        self.effect_list = list(self.effects_map.keys())

    def apply(self, roi, name):
        """Main entry point to apply an effect by name."""
        effect_func = self.effects_map.get(name, self._none)
        return effect_func(roi)

    def _none(self, roi):
        return roi
    
     #custom effects. Can really do whatever you want
    #alot of these effects are me just messing around with opencv functions to see what looks cool

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