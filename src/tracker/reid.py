"""
Lightweight ReID (Re-Identification) Module
Appearance-based re-identification using LAB color, HOG, texture, and edge features.
"""

import cv2
import numpy as np


class LightweightReID:
    """Lightweight ReID: LAB color + HOG + texture + edge density."""

    def __init__(self):
        self.feature_dim = 512

    def extract_features(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            crop = cv2.resize(crop, (128, 256))
            h = crop.shape[0]
            head = crop[: int(h * 0.3), :]
            torso = crop[int(h * 0.3) : int(h * 0.7), :]
            legs = crop[int(h * 0.7) :, :]
            feats = []
            for region in (head, torso, legs):
                feats.append(self._lab(region))
            for region in (head, torso, legs):
                feats.append(self._hog(region))
            for region in (head, torso, legs):
                feats.append(self._texture(region))
            for region in (head, torso, legs):
                feats.append(self._edge_density(region))
            features = np.concatenate(feats)
            if len(features) > self.feature_dim:
                features = features[: self.feature_dim]
            else:
                features = np.pad(features, (0, self.feature_dim - len(features)), "constant")
            features = features / (np.linalg.norm(features) + 1e-8)
            return features
        except Exception:
            return None

    def _lab(self, img):
        if img.size == 0:
            return np.zeros(64)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([lab], [0], None, [32], [0, 100])
        hist_a = cv2.calcHist([lab], [1], None, [16], [0, 255])
        hist_b = cv2.calcHist([lab], [2], None, [16], [0, 255])
        hist_l = cv2.normalize(hist_l, hist_l).flatten()
        hist_a = cv2.normalize(hist_a, hist_a).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        return np.concatenate([hist_l, hist_a, hist_b])

    def _hog(self, img):
        if img.size == 0:
            return np.zeros(64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            gray = cv2.resize(gray, (16, 16))
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        direction = ((direction + 180) % 360).astype(np.uint8)
        h, w = magnitude.shape
        cell = 8
        n_x, n_y = w // cell, h // cell
        hist = np.zeros(64)
        for i in range(0, min(n_y * cell, h), cell):
            for j in range(0, min(n_x * cell, w), cell):
                cell_mag = magnitude[i : i + cell, j : j + cell]
                cell_dir = direction[i : i + cell, j : j + cell]
                for mag, d in zip(cell_mag.flatten(), cell_dir.flatten()):
                    bin_idx = int(d / 360 * 64) % 64
                    hist[bin_idx] += mag
        hist = hist / (np.linalg.norm(hist) + 1e-8)
        return hist

    def _texture(self, img):
        if img.size == 0:
            return np.zeros(32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            gray = cv2.resize(gray, (16, 16))
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        hist = cv2.calcHist([local_var.astype(np.uint8)], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _edge_density(self, img):
        if img.size == 0:
            return np.zeros(16)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = max(1, h // grid_h), max(1, w // grid_w)
        densities = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = edges[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                density = np.sum(cell > 0) / (cell_h * cell_w)
                densities.append(density)
        densities = np.array(densities[:16])
        if len(densities) < 16:
            densities = np.pad(densities, (0, 16 - len(densities)), "constant")
        return densities

    @staticmethod
    def similarity(f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        try:
            dp = np.dot(f1, f2)
            return float(dp / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8))
        except Exception:
            return 0.0

