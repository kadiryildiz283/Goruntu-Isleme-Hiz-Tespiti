import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
# tracker.py dosyasının aynı dizinde olduğundan emin ol
from tracker import Tracker # Varsa kendi takipçi dosyanız
# Eğer tracker.py yoksa ve basit bir takipçi isterseniz,
# OpenCV'nin veya başka kütüphanelerin takipçilerini kullanmanız gerekebilir.
# Bu kodda tracker.py'nin var olduğu varsayılmıştır.
import math
import time

# --- Ayarlar ---
VIDEO_PATH = '8.mp4' # Video dosyanızın yolu
YOLO_MODEL = 'yolov8s.pt' # Kullanılacak YOLO modeli
COCO_FILE = 'coco.txt' # COCO sınıf isimleri dosyası
FRAME_PROCESS_INTERVAL = 3 # Her X karede bir işlem yap (performans için)
TARGET_CLASSES = ['car', 'truck', 'bus'] # Takip edilecek sınıflar

# --- Deneysel Hız Tahmini Ayarları ---
# Bu değerler TAHMİNİDİR ve sahneye/videoya göre ayarlanması GEREKİR!
ASSUMED_CAR_WIDTH_METERS = 1.8 # Ortalama bir araba genişliği (metre)

# Köşegen değişimine göre uzaklık ayarlama faktörleri (ÇOK DENEYSEL!)
# Eşik değerleri (piksel/saniye cinsinden köşegen değişim hızı)
DIAG_RATE_THRESHOLD_NEAR = -40 # Hızlı küçülme eşiği (yakınlaşıyor olabilir) - Daha negatif
DIAG_RATE_THRESHOLD_FAR = -5   # Yavaş küçülme eşiği (uzaklaşıyor olabilir) - Sıfıra yakın negatif
# Ayarlama Çarpanları
ADJUST_FACTOR_FAR = 1.8     # Uzaktaysa (yavaş küçülüyorsa) hızı artır
ADJUST_FACTOR_NEAR = 0.7    # Yakındaysa (hızlı küçülüyorsa) hızı azalt
# Not: Pozitif değişim (büyüme) veya sıfıra çok yakın değişim için faktör 1.0 olacak
# -------------------------------------

model = YOLO(YOLO_MODEL)
cap = cv2.VideoCapture(VIDEO_PATH)

# Video FPS'ini al
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Uyarı: Video FPS değeri alınamadı. Varsayılan 30 kullanılıyor.")
    fps = 30

# Kareler arası zaman (saniye cinsinden)
frame_time_sec = FRAME_PROCESS_INTERVAL / fps

try:
    with open(COCO_FILE, "r") as my_file:
        data = my_file.read()
        class_list = data.split("\n")
except FileNotFoundError:
    print(f"Hata: {COCO_FILE} dosyası bulunamadı.")
    exit()

count = 0
try:
    tracker = Tracker()
except NameError:
    print("Hata: Tracker sınıfı bulunamadı. tracker.py dosyasının olduğundan emin olun veya başka bir takipçi kullanın.")
    exit()


# Araçların önceki PİKSEL konumlarını, zamanlarını ve KÖŞEGENLERİNİ saklamak için
prev_pixel_coords = {}
prev_time_stamp = {}
prev_bbox_diag = {} # Önceki köşegen uzunluğunu saklamak için

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Zaman damgası
    current_time_sec = count / fps

    count += 1
    if count % FRAME_PROCESS_INTERVAL != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    original_frame = frame.copy()

    results = model.predict(frame, verbose=False) # verbose=False sessiz mod
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a.numpy()).astype("float")

    detections = []
    conf_threshold = 0.4 # Güven eşiğini ayarlayabilirsiniz
    for index, row in px.iterrows():
        conf = float(row[4])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        # Sınıf indeksinin geçerli olup olmadığını kontrol et
        if 0 <= d < len(class_list):
             c = class_list[d]
             if any(target in c for target in TARGET_CLASSES):
                 detections.append([x1, y1, x2, y2])
        else:
            # print(f"Uyarı: Geçersiz sınıf indeksi {d}")
            pass

    bbox_id = tracker.update(detections)
    current_speeds = {}

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4) # Tam sayıya çevir

        # Referans noktası (alt-orta) - Piksel koordinatları
        cx_pixel = int(x3 + x4) // 2
        cy_pixel = int(y4)
        current_pixel_coord = (cx_pixel, cy_pixel)

        # 1. Köşegen Uzaklığını Hesapla
        bbox_width_pixels = x4 - x3
        bbox_height_pixels = y4 - y3
        bbox_diag = 0
        if bbox_width_pixels > 0 and bbox_height_pixels > 0:
            bbox_diag = math.sqrt(bbox_width_pixels**2 + bbox_height_pixels**2)

        speed_kmh = 0 # Varsayılan hız
        scale_factor_m_per_pix = 0 # Tahmini ölçek (metre/piksel)
        movement_angle_deg = -1 # Hareket açısı
        distance_adjustment_factor = 1.0 # Uzaklık ayar faktörü

        # Eğer bu araç daha önce takip edildiyse hızını hesapla
        if id in prev_pixel_coords and id in prev_time_stamp:
            prev_coord_pix = prev_pixel_coords[id]
            prev_time = prev_time_stamp[id]

            # Geçen süreyi hesapla (saniye)
            elapsed_time = current_time_sec - prev_time
            # VEYA sabit kare işleme aralığı kullan:
            # elapsed_time = frame_time_sec

            if elapsed_time > 0:
                # Piksel cinsinden mesafeyi hesapla
                pixel_distance = math.dist(prev_coord_pix, current_pixel_coord)

                # Piksel cinsinden hızı hesapla (piksel/s)
                speed_pixels_per_sec = pixel_distance / elapsed_time

                # 2. Hareket Açısını Hesapla
                delta_x = current_pixel_coord[0] - prev_coord_pix[0]
                delta_y = current_pixel_coord[1] - prev_coord_pix[1] # Y aşağı doğru pozitif
                if abs(delta_x) > 0.5 or abs(delta_y) > 0.5: # Sadece küçük hareketleri filtrele
                    movement_angle_rad = math.atan2(-delta_y, delta_x) # Kartezyen açı için -delta_y
                    movement_angle_deg = math.degrees(movement_angle_rad)
                    if movement_angle_deg < 0:
                        movement_angle_deg += 360
                else:
                    movement_angle_deg = -1 # Açı hesaplanamadı (veya araç duruyor)


                # 3. Ölçek Faktörü (Genişlik & Açı ile Metre/Piksel Tahmini)
                if bbox_width_pixels > 0:
                    corrected_bbox_width = bbox_width_pixels # Varsayılan
                    if movement_angle_deg != -1:
                        # Açıya göre düzeltme faktörü (90 derecede 1, 0/180'de 0'a yakın)
                        # abs(açı - 90) -> 90 dereceden sapma
                        angle_diff_from_90 = abs(movement_angle_deg - 90)
                        # Eğer 180'den büyükse diğer taraftan sapma (örn 270 için de 90)
                        if angle_diff_from_90 > 90:
                            angle_diff_from_90 = 180 - angle_diff_from_90

                        # Kosinüs fonksiyonu (0 derecede 1, 90 derecede 0)
                        # Sapma 0 iken (açı 90) faktör 1, sapma 90 iken (açı 0/180) faktör 0
                        angle_correction_factor = math.cos(math.radians(angle_diff_from_90))

                        # Çok küçük faktörleri engelle (örn. tam yatay gidince genişlik 0 olmasın)
                        min_factor = 0.1
                        angle_correction_factor = max(min_factor, angle_correction_factor)

                        corrected_bbox_width = bbox_width_pixels * angle_correction_factor
                    else:
                        # Açı hesaplanamadıysa düzeltme yapma
                        angle_correction_factor = 1.0 # Göstermek için

                    if corrected_bbox_width > 0:
                         # Ölçek faktörünü hesapla (m/piksel)
                         scale_factor_m_per_pix = ASSUMED_CAR_WIDTH_METERS / corrected_bbox_width
                    else:
                         scale_factor_m_per_pix = 0 # Hesaplama başarısız
                else:
                    scale_factor_m_per_pix = 0 # Hesaplama başarısız


                # --- İlk Hız Tahmini ---
                speed_mps_initial = 0
                if scale_factor_m_per_pix > 0:
                    speed_mps_initial = speed_pixels_per_sec * scale_factor_m_per_pix


                # 4. Uzaklık Ayarı (Köşegen Değişimi ile - DENEYSEL)
                if id in prev_bbox_diag:
                    prev_diag = prev_bbox_diag[id]
                    if prev_diag > 0 and bbox_diag > 0: # Geçerli köşegen değerleri varsa
                        diag_change = bbox_diag - prev_diag
                        diag_change_rate = diag_change / elapsed_time # piksel/saniye

                        # Kaba eşiklere göre ayarlama faktörünü belirle
                        if diag_change_rate < DIAG_RATE_THRESHOLD_NEAR: # Hızlı küçülüyor -> Yakın
                            distance_adjustment_factor = ADJUST_FACTOR_NEAR
                        elif diag_change_rate < DIAG_RATE_THRESHOLD_FAR: # Yavaş küçülüyor -> Uzak
                            distance_adjustment_factor = ADJUST_FACTOR_FAR
                        # Diğer durumlar için (büyüyor veya az değişiyor) faktör 1.0 kalır

                        # print(f"ID: {id}, Diag Rate: {diag_change_rate:.1f}, Adj Factor: {distance_adjustment_factor}") # Debug
                else:
                    # Önceki köşegen yoksa ayarlama yapma
                    distance_adjustment_factor = 1.0


                # --- Nihai Hız Hesabı ---
                speed_mps_final = speed_mps_initial * distance_adjustment_factor
                speed_kmh = speed_mps_final * 3.6

                # Mantıksız hızları filtrele
                if speed_kmh < 0: speed_kmh = 0
                if speed_kmh > 200: speed_kmh = 0 # Veya başka bir üst limit


        # Mevcut durumu sonraki kare için sakla
        prev_pixel_coords[id] = current_pixel_coord
        prev_time_stamp[id] = current_time_sec
        if bbox_diag > 0: # Sadece geçerli köşegen varsa sakla
             prev_bbox_diag[id] = bbox_diag


        # --- Çizimler ---
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        label_id = f"ID: {id}"
        label_speed = ""
        if speed_kmh > 0:
            label_speed = f"{int(speed_kmh)} km/h"
        label_diag = f" D:{int(bbox_diag)}" if bbox_diag > 0 else ""
        # label_angle = f" A:{int(movement_angle_deg)}" if movement_angle_deg != -1 else ""
        # label_adj = f"Adj:{distance_adjustment_factor:.1f}" # Ayarlama faktörünü göster

        cv2.putText(frame, label_id, (x3, y3 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, label_speed, (x3, y3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, label_diag, (x3 + 50, y3 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.putText(frame, label_angle, (x3 + 100, y3 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.putText(frame, label_adj, (x3 + 100, y3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.circle(frame, (cx_pixel, cy_pixel), 3, (0, 255, 0), -1) # Merkez nokta

    cv2.imshow("Hız Tahmini (Deneysel V2)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
