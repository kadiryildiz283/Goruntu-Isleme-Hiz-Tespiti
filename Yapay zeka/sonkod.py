import cv2
import numpy as np
import math
from ultralytics import YOLO
import sys

# --- Ayarlar Bölümü ---
# 1. Kalibrasyon için kullanılacak resim
CALIBRATION_IMAGE_PATH = '80CM.png' 

# 2. 3D konumu ve yer değiştirmesi hesaplanacak resimler
RESIM_1_YOL = '80CM.png'      # Topun ilk konumu
RESIM_2_YOL = 'deneme1.png' # Topun ikinci konumu

# 3. Genel Ayarlar
BALL_DIAMETER_CM = 19.0 # Topun gerçek çapı (cm)
YOLO_MODEL = 'yolov8x.pt' # Kullanılacak YOLOv8 modeli
TARGET_CLASS_NAME = 'sports ball'

# --- Fonksiyon Tanımlamaları ---

def calibrate_camera(model, calib_image_path, real_obj_width_cm):
    """
    Verilen kalibrasyon görüntüsünü kullanarak kameranın odak uzaklıklarını (fx, fy)
    ve HFOV'unu hesaplar. Bu parametreleri bir sözlük olarak döndürür.
    """
    print("--- KAMERA KALİBRASYONU BAŞLATILIYOR ---")
    
    try:
        known_distance_cm = float(input(f"Lütfen kalibrasyon resmindeki ('{calib_image_path}') topun kameraya bilinen mesafesini (cm) girin: "))
        if known_distance_cm <= 0:
            raise ValueError("Mesafe pozitif olmalıdır.")
    except ValueError as e:
        print(f"Hata: Geçersiz giriş. Lütfen sayısal bir değer girin. {e}")
        return None

    calib_frame = cv2.imread(calib_image_path)
    if calib_frame is None:
        print(f"Hata: Kalibrasyon dosyası açılamadı ({calib_image_path}).")
        return None

    H, W = calib_frame.shape[:2]
    print(f"Kalibrasyon resmi: {calib_image_path}, Boyut: {W}x{H}")

    results = model(calib_frame, verbose=False, classes=[k for k, v in model.names.items() if v == TARGET_CLASS_NAME])[0]
    
    object_pixel_width = 0
    if results.boxes:
        # En geniş topu kalibrasyon nesnesi olarak kabul et
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            if width > object_pixel_width:
                object_pixel_width = float(width)

    if object_pixel_width == 0:
        print("Hata: Kalibrasyon resminde hedef nesne bulunamadı.")
        return None

    print(f"Kalibrasyon nesnesi bulundu, piksel genişliği: {object_pixel_width:.2f} piksel")

    # Odak uzaklığı (fx) hesapla: fx = (Piksel Genişliği * Mesafe) / Gerçek Genişlik
    f_x = (object_pixel_width * known_distance_cm) / real_obj_width_cm
    
    # Genellikle pikseller kare olduğundan fy = fx varsayılır. Bu daha kararlı sonuç verir.
    f_y = f_x

    # HFOV'u bilgilendirme amacıyla fx'ten geri hesapla
    hfov_rad = 2 * math.atan(W / (2 * f_x))
    hfov_deg = math.degrees(hfov_rad)

    print(f"Hesaplanan Yatay Odak Uzaklığı (f_x): {f_x:.2f} piksel")
    print(f"Hesaplanan Dikey Odak Uzaklığı (f_y): {f_y:.2f} piksel")
    print(f"Hesaplanan Yatay Görüş Açısı (HFOV): {hfov_deg:.2f} derece")
    print("--- KALİBRASYON TAMAMLANDI ---\n")
    
    return {"f_x": f_x, "f_y": f_y, "W": W, "H": H, "hfov_deg": hfov_deg}

def get_ball_3d_coordinates(model, image_path, ball_diameter_cm, camera_params):
    """
    Kalibrasyondan elde edilen kamera parametrelerini kullanarak
    bir görüntüdeki topun 3D (x, y, z) koordinatlarını hesaplar.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Hata: Görüntü yüklenemedi -> {image_path}")
        return None

    # Kamera parametrelerini sözlükten al
    f_x, f_y, W, H = camera_params['f_x'], camera_params['f_y'], camera_params['W'], camera_params['H']

    results = model(frame, verbose=False, classes=[k for k, v in model.names.items() if v == TARGET_CLASS_NAME])[0]
    
    best_box = None
    max_width = 0
    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            if width > max_width:
                max_width = width
                best_box = (x1, y1, x2, y2)
    
    if best_box is None:
        print(f"Uyarı: '{image_path}' içinde top bulunamadı.")
        return None

    x1, y1, x2, y2 = best_box
    ball_pixel_width = float(x2 - x1)
    
    # 'z' koordinatını (mesafe) hesapla
    z_cm = (ball_diameter_cm * f_x) / ball_pixel_width
    
    # 'x' koordinatını hesapla
    ball_center_x_px = (x1 + x2) / 2
    x_cm = ((ball_center_x_px - W / 2) * z_cm) / f_x
    
    # 'y' koordinatını hesapla
    ball_center_y_px = (y1 + y2) / 2
    y_cm = -((ball_center_y_px - H / 2) * z_cm) / f_y
    
    return np.array([x_cm, y_cm, z_cm])

def visualize_top_down_view(pos1, pos2, displacement, hfov_deg):
    """
    Hesaplanan konumları ve yer değiştirmeyi üstten görünümde görselleştirir.
    """
    CANVAS_SIZE = (800, 800)
    PIXELS_PER_CM = 2.0
    canvas = np.full((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), 255, dtype=np.uint8)
    origin_px = (CANVAS_SIZE[1] // 2, 50)

    # Eksenler ve Kamera ikonu
    cv2.line(canvas, (origin_px[0], 0), (origin_px[0], CANVAS_SIZE[0]), (200, 200, 200), 1)
    cv2.putText(canvas, "Z Ekseni (cm)", (origin_px[0] + 10, CANVAS_SIZE[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.circle(canvas, origin_px, 10, (0, 0, 0), -1)
    cv2.putText(canvas, "Kamera", (origin_px[0] - 30, origin_px[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    p1_px = (int(origin_px[0] + pos1[0] * PIXELS_PER_CM), int(origin_px[1] + pos1[2] * PIXELS_PER_CM))
    p2_px = (int(origin_px[0] + pos2[0] * PIXELS_PER_CM), int(origin_px[1] + pos2[2] * PIXELS_PER_CM))

    cv2.arrowedLine(canvas, p1_px, p2_px, (255, 0, 0), 2, tipLength=0.05)
    cv2.circle(canvas, p1_px, 8, (0, 165, 255), -1); cv2.circle(canvas, p1_px, 8, (0, 0, 0), 2)
    cv2.putText(canvas, "Pos1", (p1_px[0] + 15, p1_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.circle(canvas, p2_px, 8, (0, 255, 0), -1); cv2.circle(canvas, p2_px, 8, (0, 0, 0), 2)
    cv2.putText(canvas, "Pos2", (p2_px[0] + 15, p2_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Bilgilendirme metinleri (Dinamik HFOV dahil)
    cv2.putText(canvas, f"Dinamik HFOV: {hfov_deg:.2f} derece", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    cv2.putText(canvas, f"Toplam Yer Degistirme: {displacement:.1f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(canvas, f"Pos1 (x,z): ({pos1[0]:.1f}, {pos1[2]:.1f}) cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(canvas, f"Pos2 (x,z): ({pos2[0]:.1f}, {pos2[2]:.1f}) cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
    
    return canvas

# --- Ana Betik Akışı ---
if __name__ == "__main__":
    print("YOLO modeli yükleniyor...")
    try:
        model = YOLO(YOLO_MODEL)
    except Exception as e:
        print(f"Hata: YOLO modeli yüklenemedi. {e}"); sys.exit()

    # 1. Adım: Kamerayı kalibre et
    camera_parameters = calibrate_camera(model, CALIBRATION_IMAGE_PATH, BALL_DIAMETER_CM)
    if not camera_parameters:
        print("Kalibrasyon başarısız. Program sonlandırılıyor."); sys.exit()

    # 2. Adım: Her iki görüntü için 3D koordinatları hesapla
    print("1. Görüntü için 3D koordinatlar hesaplanıyor...")
    pos1_3d = get_ball_3d_coordinates(model, RESIM_1_YOL, BALL_DIAMETER_CM, camera_parameters)

    print("2. Görüntü için 3D koordinatlar hesaplanıyor...")
    pos2_3d = get_ball_3d_coordinates(model, RESIM_2_YOL, BALL_DIAMETER_CM, camera_parameters)

    if pos1_3d is None or pos2_3d is None:
        print("\nBir veya daha fazla görüntüde top tespit edilemedi. Program sonlandırılıyor."); sys.exit()

    # 3. Adım: Yer değiştirmeyi hesapla
    total_displacement = np.linalg.norm(pos2_3d - pos1_3d)

    # 4. Adım: Sonuçları yazdır
    print("\n--- HESAPLAMA SONUÇLARI ---")
    print(f"Topun İlk Konumu (x, y, z): ({pos1_3d[0]:.1f}, {pos1_3d[1]:.1f}, {pos1_3d[2]:.1f}) cm")
    print(f"Topun İkinci Konumu (x, y, z): ({pos2_3d[0]:.1f}, {pos2_3d[1]:.1f}, {pos2_3d[2]:.1f}) cm")
    print("-" * 30)
    print(f"Toplam 3D Yer Değiştirme: {total_displacement:.2f} cm")
    print("-----------------------------\n")

    # 5. Adım: Görselleştir
    print("Üstten görünüm görselleştirmesi oluşturuluyor...")
    visualization = visualize_top_down_view(pos1_3d, pos2_3d, total_displacement, camera_parameters['hfov_deg'])

    cv2.imshow("Topun 3D Yer Degistirmesi (Dinamik Kalibrasyonlu)", visualization)
    print("Sonuç penceresi açıldı. Kapatmak için herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")
