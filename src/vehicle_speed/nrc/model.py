r"""Compact Multi-Layer Perceptron (MLP) for Neural Residual Calibration (NRC).

Architecture: [5] -> [32] -> [16] -> [8] -> [1]
Predicts the residual speed error delta_v such that:
$v_{\text{final}} = \max(0, v_{\text{raw}} + \Delta v)$
Zero-dependency NumPy inference (< 0.05 ms latency on CPU).
"""

from pathlib import Path
from typing import Any

import numpy as np

from vehicle_speed.nrc.features import NRCFeatures


class NeuralResidualCalibrator:
    r"""5 girdili 1 çıktılı Nöral Artık Kalibratör (NRC).

    Ağ Mimarisi:
        Girdi (5) -> Dense(32) + LeakyReLU
                  -> Dense(16) + LeakyReLU
                  -> Dense(8)  + LeakyReLU
                  -> Dense(1)  -> \Delta v (m/s)
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        """NRC model ilklendirici.

        Args:
            weights_path: Eğitilmiş .npz model ağırlık dosya yolu (opsiyonel).
        """
        # Katman boyutları: [5, 32, 16, 8, 1]
        self.layer_sizes = [5, 32, 16, 8, 1]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        if weights_path is not None and weights_path.is_file():
            self.load_weights(weights_path)
        else:
            self._init_default_weights()

    def _init_default_weights(self) -> None:
        """He/Kaiming normalizasyonu ile varsayılan ağırlıkları başlatır."""
        # Deterministik tohum
        rng = np.random.default_rng(seed=42)
        self.weights = []
        self.biases = []

        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True):
            # He / Kaiming standardı
            std = np.sqrt(2.0 / in_dim)
            w = np.asarray(rng.normal(0.0, std, size=(in_dim, out_dim)), dtype=np.float32)
            # Çıkış katmanında başlangıç artık düzeltmesini sıfıra yakın başlat
            if out_dim == 1:
                w = w * 0.01
            b = np.zeros((out_dim,), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _leaky_relu(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """LeakyReLU aktivasyon fonksiyonu."""
        return np.where(x > 0, x, x * alpha)

    def predict_residual(self, features: NRCFeatures) -> float:
        r"""5 boyutlu öznitelik vektöründen artık hız hatasını ($\Delta v$) tahmin eder.

        Args:
            features: 5 elemanlı NRCFeatures nesnesi.

        Returns:
            float: Metre/saniye cinsinden düzeltme miktarı ($\Delta v$).
        """
        x = features.to_numpy().reshape(1, -1)

        # İleri geçiş (Forward Pass)
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = (x @ w) + b
            if i < len(self.weights) - 1:
                x = self._leaky_relu(x)

        return float(x[0, 0])

    def calibrate(self, raw_speed_ms: float, features: NRCFeatures) -> float:
        r"""Ham hızı artık hata ile kalibre eder.

        Matematiksel kural:
            $v_{\text{calibrated}} = \max(0.0, v_{\text{raw}} + \Delta v)$

        Args:
            raw_speed_ms: Ham hız (m/s).
            features: Giriş öznitelikleri.

        Returns:
            float: Kalibre edilmiş nihai hız (m/s).
        """
        delta_v = self.predict_residual(features)
        # Fiziksel hız negatif olamaz
        return max(0.0, raw_speed_ms + delta_v)

    def save_weights(self, path: Path) -> None:
        """Model ağırlıklarını .npz dosya formatında kaydeder."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, Any] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            tensors[f"w_{i}"] = w
            tensors[f"b_{i}"] = b
        np.savez(path, **tensors)

    def load_weights(self, path: Path) -> None:
        """Ağırlıkları .npz dosyasından yükler."""
        data = np.load(path)
        self.weights = []
        self.biases = []
        num_layers = len(self.layer_sizes) - 1
        for i in range(num_layers):
            self.weights.append(data[f"w_{i}"].astype(np.float32))
            self.biases.append(data[f"b_{i}"].astype(np.float32))
