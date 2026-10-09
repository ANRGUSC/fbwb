"""Upstream optical-flow features adapted to Webots BGRA frames."""
import cv2
import numpy as np


class OpticFlow:
    def __init__(self, noise_floor=0.01):
        self.previous = None
        self.noise_floor = noise_floor

    def compute(self, bgra):
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError('Expected H x W x 4 BGRA camera frame')
        gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
        gray = cv2.resize(gray, (160, 120))
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.previous is None:
            self.previous = gray
            return np.zeros(4, dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(self.previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.previous = gray
        # The upstream ratios amplify numerical flow on identical images to 1.
        # Suppress sub-pixel noise before normalizing (pixels per 20 ms frame).
        if np.mean(np.linalg.norm(flow, axis=2)) < self.noise_floor:
            return np.zeros(4, dtype=np.float32)
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        y, x = np.mgrid[:h, :w]
        radial = (flow[:, :, 0] * (x-cx) + flow[:, :, 1] * (y-cy)) / (np.hypot(x-cx, y-cy) + 1e-6)
        forward = np.clip(np.mean(radial) * 0.5, 0, 1)
        left = np.mean(np.abs(flow[:, :cx, 0]))
        right = np.mean(np.abs(flow[:, cx:, 0]))
        top = np.mean(np.abs(flow[:cy, :, 1]))
        bottom = np.mean(np.abs(flow[cy:, :, 1]))
        return np.array([forward, np.clip(left / (left+right+1e-6)*2, 0, 1), np.clip(right / (left+right+1e-6)*2, 0, 1), np.clip(bottom / (top+bottom+1e-6)*2, 0, 1)], dtype=np.float32)
