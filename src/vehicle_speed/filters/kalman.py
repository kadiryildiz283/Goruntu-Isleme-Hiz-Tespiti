"""3D Constant Velocity Kalman Filter for spatial trajectory and velocity estimation.

Implements linear state estimation for 3D coordinates [x, y, z] and velocities [vx, vy, vz].
Attenuates optical measurement jitter and handles matrix inversion stability (Fail-Safe).
"""

import math

import numpy as np


class KalmanFilter3D:
    r"""3 Boyutlu Sabit Hızlı (Constant Velocity) Kalman Filtresi.

    Durum vektörü:
        $\mathbf{x} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^T \in \mathbb{R}^6$

    Ölçüm vektörü:
        $\mathbf{z} = [x, y, z]^T \in \mathbb{R}^3$

    Fiziksel Model:
        $\mathbf{x}_{k} = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{w}_k$
        $\mathbf{z}_{k} = \mathbf{H} \mathbf{x}_{k} + \mathbf{v}_k$
    """

    def __init__(
        self,
        initial_position: tuple[float, float, float],
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.8,
    ) -> None:
        """KalmanFilter3D başlatıcı.

        Args:
            initial_position: Başlangıç (X, Y, Z) metre konumu.
            process_noise_std: Süreç gürültüsü standart sapması ($q$).
            measurement_noise_std: Ölçüm gürültüsü standart sapması ($r$).
        """
        # Durum vektörü [x, y, z, vx, vy, vz]
        x0, y0, z0 = initial_position
        self.state: np.ndarray = np.array([x0, y0, z0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Durum kovaryans matrisi P (6x6)
        self.P: np.ndarray = np.eye(6, dtype=np.float64)
        # Hız belirsizliği başlangıçta daha yüksektir
        self.P[3:, 3:] = self.P[3:, 3:] * 5.0

        # Ölçüm matrisi H (3x6) - Yalnızca konum ölçülür
        self.H: np.ndarray = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Ölçüm gürültü kovaryansı R (3x3)
        self.R: np.ndarray = np.multiply(np.eye(3, dtype=np.float64), measurement_noise_std**2)

        # Temel süreç gürültü ölçeği
        self.q_std = process_noise_std
        self.eps = 1e-6

    def predict(self, dt: float) -> np.ndarray:
        r"""Zaman adımıyla durum ve kovaryans tahmini yapar.

        Matematiksel geçiş:
            F = [[I_3, dt * I_3],
                 [0_3,      I_3]]

        Args:
            dt: İki adım arasındaki zaman farkı (saniye, $\Delta t > 0$).

        Returns:
            np.ndarray: Tahmin edilen 6 elemanlı durum vektörü.

        Raises:
            ValueError: dt <= 0 durumunda.
        """
        if dt <= 0.0:
            msg = f"dt pozitif olmalıdır, alınan: {dt}"
            raise ValueError(msg)

        # Durum geçiş matrisi F
        F: np.ndarray = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Süreç gürültü kovaryansı Q (Sürekli beyaz gürültü ivme modeli)
        dt2 = 0.5 * (dt**2)
        G: np.ndarray = np.array(
            [
                [dt2, 0.0, 0.0],
                [0.0, dt2, 0.0],
                [0.0, 0.0, dt2],
                [dt, 0.0, 0.0],
                [0.0, dt, 0.0],
                [0.0, 0.0, dt],
            ],
            dtype=np.float64,
        )
        Q: np.ndarray = np.multiply(G @ G.T, self.q_std**2)

        # Durum ve kovaryans güncelleme
        self.state = F @ self.state
        self.P = (F @ self.P @ F.T) + Q

        return self.state

    def update(self, measurement: tuple[float, float, float]) -> np.ndarray:
        r"""Optik Pinhole modelinden gelen (X, Y, Z) ölçümüyle durumu günceller.

        Matematiksel güncelleme:
            $\mathbf{y} = \mathbf{z} - \mathbf{H} \mathbf{x}$
            $\mathbf{S} = \mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R}$
            $\mathbf{K} = \mathbf{P} \mathbf{H}^T \mathbf{S}^{-1}$
            $\mathbf{x} = \mathbf{x} + \mathbf{K} \mathbf{y}$
            $\mathbf{P} = (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}$

        Args:
            measurement: Ölçülen (x, y, z) metre konumu.

        Returns:
            np.ndarray: Düzeltilmiş 6 elemanlı durum vektörü.
        """
        z: np.ndarray = np.array(measurement, dtype=np.float64)
        y: np.ndarray = z - (self.H @ self.state)  # Yenilik (Innovation)

        S: np.ndarray = (self.H @ self.P @ self.H.T) + self.R
        # Kovaryans tekilleşme önlemi (regularization)
        S = S + (np.eye(3, dtype=np.float64) * self.eps)

        K: np.ndarray = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman kazancı

        self.state = self.state + (K @ y)
        I_KH: np.ndarray = np.eye(6, dtype=np.float64) - (K @ self.H)
        # Joseph stabilite biçimi: P = (I-KH)P(I-KH)^T + KRK^T
        self.P = (I_KH @ self.P @ I_KH.T) + (K @ self.R @ K.T)

        return self.state

    @property
    def position(self) -> tuple[float, float, float]:
        """Anlık filtrelenmiş 3B konum (X, Y, Z) metre."""
        return float(self.state[0]), float(self.state[1]), float(self.state[2])

    @property
    def velocity(self) -> tuple[float, float, float]:
        """Anlık filtrelenmiş 3B hız (vx, vy, vz) m/s."""
        return float(self.state[3]), float(self.state[4]), float(self.state[5])

    @property
    def speed_ms(self) -> float:
        r"""3B hız vektörünün büyüklüğü: $v = \sqrt{v_x^2 + v_y^2 + v_z^2}$."""
        vx, vy, vz = self.velocity
        return math.sqrt(vx**2 + vy**2 + vz**2)
