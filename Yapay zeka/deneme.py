# Gerekli kütüphaneleri içe aktarır:
import cv2
import numpy as np
from ultralytics import YOLO
import math
# time kütüphanesi artık gerekli değil (sürekli işleme olmadığı için)
# from collections import defaultdict, deque # deque artık gerekli değil

# --- Ayarlar Bölümü ---
IMAGE_PATH = 'deneme3.png'  # İşlenecek fotoğraf dosyasının yolu.
YOLO_MODEL = 'yolov8x.pt' # Kullanılacak YOLOv8 modelinin dosya adı.

TARGET_CLASS_NAME = 'sports ball' # Hedeflenecek sınıfın adı

BALL_DIAMETER_CM = 19 # Örnek bir top çapı (cm) - Kendi topunuzun çapını girin
HFOV_DEGREES = 60.14 # Kameranın yatay görüş açısı (derece)
# --------------------

try:
    model = YOLO(YOLO_MODEL)
    MODEL_CLASSES = model.names
    TARGET_CLASSES_IDX = [k for k, v in MODEL_CLASSES.items() if v.lower() == TARGET_CLASS_NAME.lower()]
    if not TARGET_CLASSES_IDX:
        print(f"Uyarı: Hedef sınıf '{TARGET_CLASS_NAME}' modelde bulunamadı. Lütfen sınıf adını veya ID'sini kontrol edin.")
        # Bu durumda, kod ilk güvenilir tespiti (confidence > 0.5) top olarak varsayacaktır.
except Exception as e:
    print(f"Hata: YOLO modeli yüklenemedi ({YOLO_MODEL}). Hata: {e}")
    exit()

# Fotoğrafı yükle
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"Hata: Fotoğraf dosyası açılamadı veya bulunamadı ({IMAGE_PATH}).")
    exit()

frame_height, frame_width = frame.shape[:2]
print(f"Fotoğraf: {IMAGE_PATH}, Boyut: {frame_width}x{frame_height}")
if TARGET_CLASSES_IDX:
    print(f"Hedeflenen Sınıf ID(leri): {TARGET_CLASSES_IDX} ({TARGET_CLASS_NAME})")
else:
    print(f"Uyarı: '{TARGET_CLASS_NAME}' sınıfı modelde bulunamadığından, ilk güvenilir tespitler top olarak değerlendirilecektir.")
print(f"Top Çapı: {BALL_DIAMETER_CM} cm, HFOV: {HFOV_DEGREES} derece")

# Piksel başına düşen açısal çözünürlük (derece/piksel)
if frame_width > 0:
    angular_resolution_deg_per_pixel = HFOV_DEGREES / frame_width
    print(f"Açısal Çözünürlük: {angular_resolution_deg_per_pixel:.5f} derece/piksel")
else:
    print("Hata: Frame genişliği 0, açısal çözünürlük hesaplanamıyor.")
    exit()

# --- Mesafe Hesaplama Fonksiyonu ---
def calculate_distance_cm(object_pixel_width, actual_object_width_cm, global_angular_resolution_deg_per_pixel):
    if object_pixel_width <= 0 or actual_object_width_cm <= 0 or global_angular_resolution_deg_per_pixel <= 0:
        return None

    theta_object_degrees = object_pixel_width * global_angular_resolution_deg_per_pixel

    if theta_object_degrees <= 0 or theta_object_degrees >= 180:
        return None

    try:
        angle_rad_half = math.radians(theta_object_degrees / 2.0)
        if math.tan(angle_rad_half) == 0:
            return None
        distance = actual_object_width_cm / (2.0 * math.tan(angle_rad_half))
    except (ZeroDivisionError, ValueError):
        return None
    return distance

# Fotoğrafı işle
results = model(frame, verbose=False, classes=TARGET_CLASSES_IDX if TARGET_CLASSES_IDX else None)[0]
annotated_frame = frame.copy()
detection_id_counter = 0

if results.boxes is not None:
    for i in range(len(results.boxes.xyxy)):
        xyxy = results.boxes.xyxy[i].tolist()
        confidence = results.boxes.conf[i].item()
        class_id = int(results.boxes.cls[i].item())

        if (TARGET_CLASSES_IDX and class_id in TARGET_CLASSES_IDX) or \
           (not TARGET_CLASSES_IDX and confidence > 0.01):
            detection_id_counter += 1
            x1, y1, x2, y2 = map(int, xyxy)
            current_ball_pixel_width = float(x2 - x1)

            current_distance_cm = calculate_distance_cm(
                current_ball_pixel_width,
                BALL_DIAMETER_CM,
                angular_resolution_deg_per_pixel
            )

            distance_str = "Mesafe: N/A"
            if current_distance_cm is not None and current_distance_cm > 0:
                distance_str = f"Mesafe: {current_distance_cm:.1f} cm"

            label_parts = [f"Top {detection_id_counter}", distance_str]

            box_color = (0, 255, 0) # Yeşil
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

            label_y_start_pos = y1 - 7
            if y1 < 30: label_y_start_pos = y2 + 20

            for i, part_text in enumerate(label_parts):
                text_y_pos = label_y_start_pos + (i * 18) if y1 < 30 else label_y_start_pos - (i * 18)
                # Beyaz arka planlı metin (daha iyi okunabilirlik için)
                cv2.putText(annotated_frame, part_text, (x1 + 3, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                # Asıl renkli metin
                cv2.putText(annotated_frame, part_text, (x1 + 3, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

# Genel bilgileri ekrana yazdır
cv2.putText(annotated_frame, f"HFOV: {HFOV_DEGREES:.2f}deg, BallDia: {BALL_DIAMETER_CM:.1f}cm", (10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)


# --- KONTROL NOKTASI: İŞLENMİŞ GÖRÜNTÜYÜ KAYDET ---
print(f"KAYIT ÖNCESİ - annotated_frame Boyutları: {annotated_frame.shape}")
print(f"KAYIT ÖNCESİ - annotated_frame Veri Tipi: {annotated_frame.dtype}")
kayit_basarili = cv2.imwrite("debug_output.png", annotated_frame)
if kayit_basarili:
    print("DEBUG: 'debug_output.png' dosyası başarıyla kaydedildi. Lütfen bu dosyayı kontrol edin.")
else:
    print("DEBUG: 'debug_output.png' dosyası KAYDEDİLEMEDİ.")
# --- KONTROL NOKTASI SONU ---

# Sonucu göster
cv2.imshow("Top Mesafe Tespiti (Fotoğraf)", annotated_frame)
print("Sonucu görmek için bir tuşa basın ve kapatmak için ESC tuşuna basın...")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("İşlem tamamlandı.")

# Sonucu göster
cv2.imshow("Top Mesafe Tespiti (Fotoğraf)", annotated_frame)
print("Sonucu görmek için bir tuşa basın ve kapatmak için ESC tuşuna basın...")
cv2.waitKey(0) # Bir tuşa basılana kadar bekle
cv2.destroyAllWindows()
print("İşlem tamamlandı.")
