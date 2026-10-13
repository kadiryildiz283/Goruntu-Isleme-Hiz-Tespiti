"""Unit tests for vehicle_speed.geometry module."""

import math

import pytest

from vehicle_speed.geometry import (
    estimate_depth,
    focal_length_from_hfov,
    focal_length_from_reference,
    project_3d,
)


@pytest.mark.parametrize(
    ("f_px", "real_width_m", "pixel_width_px", "expected_m"),
    [
        (1200.0, 0.52, 100.0, 6.24),
        (1200.0, 1.50, 100.0, 18.0),
        (1200.0, 0.52, 10.0, 62.4),
        (600.0, 0.52, 100.0, 3.12),
        (1000.0, 0.185, 100.0, 1.85),
    ],
)
def test_estimate_depth_matches_pinhole_formula(
    f_px: float, real_width_m: float, pixel_width_px: float, expected_m: float
) -> None:
    """Pinhole kamera formülüyle derinlik hesabı doğrulanır: Z = (f * W) / w."""
    assert estimate_depth(f_px, real_width_m, pixel_width_px) == pytest.approx(expected_m)


def test_estimate_depth_is_inversely_proportional_to_pixel_width() -> None:
    """Z ~ 1/P ilişkisi: piksel genişliği yarıya inince derinlik iki katına çıkar."""
    near = estimate_depth(1200.0, 0.52, 200.0)
    far = estimate_depth(1200.0, 0.52, 100.0)
    assert far == pytest.approx(2.0 * near)


@pytest.mark.parametrize("bad_pixel_w", [0.0, -1.0, -100.5])
def test_estimate_depth_rejects_non_positive_pixel_width(bad_pixel_w: float) -> None:
    """Sıfır veya negatif piksel genişliğinde Fail-Closed ValueError yükseltilir."""
    with pytest.raises(ValueError, match=r"Piksel genişliği.*pozitif olmalıdır"):
        estimate_depth(1200.0, 0.52, bad_pixel_w)


@pytest.mark.parametrize("bad_focal", [0.0, -1200.0])
def test_estimate_depth_rejects_non_positive_focal_length(bad_focal: float) -> None:
    """Sıfır veya negatif odak uzaklığında ValueError yükseltilir."""
    with pytest.raises(ValueError, match=r"Odak uzaklığı.*pozitif olmalıdır"):
        estimate_depth(bad_focal, 0.52, 100.0)


@pytest.mark.parametrize("bad_width", [0.0, -0.52])
def test_estimate_depth_rejects_non_positive_real_width(bad_width: float) -> None:
    """Sıfır veya negatif fiziksel genişlikte ValueError yükseltilir."""
    with pytest.raises(ValueError, match=r"Gerçek fiziksel genişlik.*pozitif olmalıdır"):
        estimate_depth(1200.0, bad_width, 100.0)


def test_pixel_jitter_error_grows_quadratically_with_depth() -> None:
    r"""±1 px titremenin derinlik hatası Z^2 ile büyür (\delta Z \propto Z^2 \delta w).

    Bu test makalenin teorik çekirdeği olan Z^2 hata yayılımını doğrular.
    """
    f_px = 1200.0
    real_width_m = 0.52
    jitter_px = 1.0

    errors: dict[float, float] = {}
    for pixel_width_px in (200.0, 100.0, 50.0, 25.0):
        nominal = estimate_depth(f_px, real_width_m, pixel_width_px)
        jittered = estimate_depth(f_px, real_width_m, pixel_width_px + jitter_px)
        errors[pixel_width_px] = abs(jittered - nominal)

    # Piksel genişliği yarıya indikçe mesafe iki katına çıkar, hata ~4 katına (Z^2) çıkmalıdır
    ratios = [
        errors[100.0] / errors[200.0],
        errors[50.0] / errors[100.0],
        errors[25.0] / errors[50.0],
    ]
    for ratio in ratios:
        assert 3.5 < ratio < 4.5, f"Beklenen ~4x büyüme (Z^2), elde edilen: {ratio:.2f}"


def test_project_3d_center_pixel_yields_zero_x_y() -> None:
    """Optik merkezdeki (cx, cy) pikselin 3B izdüşümünde X=0 ve Y=0 olmalıdır."""
    x, y, z = project_3d(
        px_x=640.0,
        px_y=360.0,
        z_m=10.0,
        f_px=1000.0,
        optical_center=(640.0, 360.0),
    )
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(10.0)


def test_project_3d_scales_linearly_with_offset() -> None:
    """Optik merkezden piksel uzaklığı ile X metre konumu doğrusal orantılıdır."""
    x1, _y1, _z1 = project_3d(740.0, 360.0, 10.0, 1000.0, (640.0, 360.0))
    x2, _y2, _z2 = project_3d(840.0, 360.0, 10.0, 1000.0, (640.0, 360.0))
    assert x1 == pytest.approx(1.0)
    assert x2 == pytest.approx(2.0)


def test_focal_length_from_reference_roundtrips() -> None:
    """Referans nesneden kalibre edilen f_px ile geri derinlik kestirimi eşleşmelidir."""
    known_dist_m = 5.0
    real_w_m = 0.52
    px_w = 150.0

    f_px = focal_length_from_reference(px_w, known_dist_m, real_w_m)
    recovered_z = estimate_depth(f_px, real_w_m, px_w)
    assert recovered_z == pytest.approx(known_dist_m)


def test_focal_length_from_hfov_matches_trigonometry() -> None:
    """HFOV'dan odak uzaklığı türetimi trigonometrik formüle uyar."""
    hfov_rad = math.radians(85.0)
    width_px = 1920
    expected_f = (1920 / 2.0) / math.tan(hfov_rad / 2.0)
    derived_f = focal_length_from_hfov(hfov_rad, width_px)
    assert derived_f == pytest.approx(expected_f)


@pytest.mark.parametrize("bad_hfov", [-0.1, 0.0, math.pi, math.pi + 0.5])
def test_focal_length_from_hfov_rejects_invalid_angles(bad_hfov: float) -> None:
    """Geçersiz HFOV radyan açılarında ValueError yükseltilir."""
    with pytest.raises(ValueError, match="HFOV açısı"):
        focal_length_from_hfov(bad_hfov, 1920)


def test_focal_length_from_hfov_rejects_non_positive_width() -> None:
    """Görüntü genişliği pozitif tamsayı olmalıdır."""
    with pytest.raises(ValueError, match="Görüntü genişliği"):
        focal_length_from_hfov(math.radians(85.0), 0)


@pytest.mark.parametrize("bad_focal", [0.0, -100.0])
def test_project_3d_rejects_non_positive_focal(bad_focal: float) -> None:
    """project_3d f_px <= 0 durumunda ValueError fırlatır."""
    with pytest.raises(ValueError, match="Odak uzaklığı pozitif"):
        project_3d(100.0, 100.0, 10.0, bad_focal, (640.0, 360.0))


def test_project_3d_rejects_negative_depth() -> None:
    """project_3d z_m < 0 durumunda ValueError fırlatır."""
    with pytest.raises(ValueError, match="Derinlik negatif olamaz"):
        project_3d(100.0, 100.0, -5.0, 1000.0, (640.0, 360.0))


@pytest.mark.parametrize(
    ("pw", "dist", "rw"),
    [
        (-1.0, 5.0, 0.52),
        (100.0, -5.0, 0.52),
        (100.0, 5.0, -0.52),
    ],
)
def test_focal_length_from_reference_rejects_non_positive_inputs(
    pw: float, dist: float, rw: float
) -> None:
    """focal_length_from_reference negatif veya sıfır girdilerde ValueError fırlatır."""
    with pytest.raises(ValueError, match="Tüm kalibrasyon girdileri"):
        focal_length_from_reference(pw, dist, rw)
