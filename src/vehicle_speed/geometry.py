"""Pure mathematical optical and spatial geometry functions.

Implements pinhole camera model calculations, coordinate system projections,
and focal length derivations without third-party dependencies (strictly no cv2/torch).
"""

import math


def estimate_depth(f_px: float, real_width_m: float, pixel_width_px: float) -> float:
    r"""Pinhole kamera modeliyle nesnenin kameraya olan eksenel derinliğini (Z) hesaplar.

    Matematiksel formül:
        Z = \frac{f_{px} \cdot W_{real}}{w_{px}}

    Args:
        f_px: Kameranın piksel cinsinden odak uzaklığı (focal length).
        real_width_m: Nesnenin bilinen fiziksel genişliği (metre).
        pixel_width_px: Nesnenin görüntü düzlemindeki genişliği (piksel).

    Returns:
        float: Nesnenin metre cinsinden derinliği ($Z$).

    Raises:
        ValueError: Girdilerden herhangi biri sıfır veya negatif olduğunda (Fail-Closed).
    """
    if f_px <= 0.0:
        msg = f"Odak uzaklığı (f_px) pozitif olmalıdır, alınan: {f_px}"
        raise ValueError(msg)
    if real_width_m <= 0.0:
        msg = f"Gerçek fiziksel genişlik (real_width_m) pozitif olmalıdır, alınan: {real_width_m}"
        raise ValueError(msg)
    if pixel_width_px <= 0.0:
        msg = f"Piksel genişliği (pixel_width_px) pozitif olmalıdır, alınan: {pixel_width_px}"
        raise ValueError(msg)

    return (f_px * real_width_m) / pixel_width_px


def project_3d(
    px_x: float,
    px_y: float,
    z_m: float,
    f_px: float,
    optical_center: tuple[float, float],
) -> tuple[float, float, float]:
    r"""2B görüntü düzlemindeki piksel koordinatını 3B kamera uzayına izdüşürür.

    Matematiksel formül:
        X = \frac{(u - c_x) \cdot Z}{f}
        Y = \frac{(v - c_y) \cdot Z}{f}
        Z = Z

    Args:
        px_x: Görüntüdeki yatay piksel koordinatı ($u$).
        px_y: Görüntüdeki düşey piksel koordinatı ($v$).
        z_m: Nesnenin metre cinsinden derinliği ($Z$).
        f_px: Odak uzaklığı (piksel).
        optical_center: Optik merkez $(c_x, c_y)$ piksel koordinatı.

    Returns:
        tuple[float, float, float]: Kamera koordinat sisteminde $(X, Y, Z)$ metre.

    Raises:
        ValueError: f_px <= 0 veya z_m < 0 durumunda.
    """
    if f_px <= 0.0:
        msg = f"Odak uzaklığı pozitif olmalıdır, alınan: {f_px}"
        raise ValueError(msg)
    if z_m < 0.0:
        msg = f"Derinlik negatif olamaz, alınan: {z_m}"
        raise ValueError(msg)

    c_x, c_y = optical_center
    x_m = ((px_x - c_x) * z_m) / f_px
    y_m = ((px_y - c_y) * z_m) / f_px
    return (x_m, y_m, z_m)


def focal_length_from_hfov(hfov_rad: float, image_width_px: int) -> float:
    r"""Yatay görüş açısından (HFOV) odak uzaklığını (f_px) türetir.

    Matematiksel formül:
        f = \frac{W_{img} / 2}{\tan(HFOV / 2)}

    Args:
        hfov_rad: Radyan cinsinden yatay görüş açısı ($0 < HFOV < \pi$).
        image_width_px: Görüntü genişliği (piksel).

    Returns:
        float: Piksel cinsinden odak uzaklığı.

    Raises:
        ValueError: hfov_rad <= 0 veya >= pi veya image_width_px <= 0 ise.
    """
    if not (0.0 < hfov_rad < math.pi):
        msg = f"HFOV açısı (0, pi) aralığında olmalıdır, alınan radyan: {hfov_rad}"
        raise ValueError(msg)
    if image_width_px <= 0:
        msg = f"Görüntü genişliği pozitif tamsayı olmalıdır, alınan: {image_width_px}"
        raise ValueError(msg)

    return (image_width_px / 2.0) / math.tan(hfov_rad / 2.0)


def focal_length_from_reference(
    pixel_width_px: float,
    known_distance_m: float,
    real_width_m: float,
) -> float:
    r"""Bilinen referans nesne mesafesinden dinamik odak uzaklığı kalibre eder.

    Matematiksel formül:
        f = \frac{w_{px} \cdot Z_{known}}{W_{real}}

    Args:
        pixel_width_px: Referans nesnenin görüntüdeki piksel genişliği.
        known_distance_m: Referans nesnenin kameraya bilinen mesafesi (metre).
        real_width_m: Referans nesnenin gerçek fiziksel genişliği (metre).

    Returns:
        float: Piksel cinsinden kalibre edilmiş odak uzaklığı.

    Raises:
        ValueError: Girdilerden herhangi biri sıfır veya negatif ise.
    """
    if pixel_width_px <= 0.0 or known_distance_m <= 0.0 or real_width_m <= 0.0:
        msg = (
            f"Tüm kalibrasyon girdileri pozitif olmalıdır: "
            f"pixel_w={pixel_width_px}, dist={known_distance_m}, real_w={real_width_m}"
        )
        raise ValueError(msg)

    return (pixel_width_px * known_distance_m) / real_width_m
