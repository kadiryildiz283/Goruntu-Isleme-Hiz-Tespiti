import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import supervision as sv # ByteTrack ve diğer yardımcı fonksiyonlar için
import math
import time
from collections import defaultdict, deque # Önceki koordinatları saklamak için

# --- Ayarlar ---
VIDEO_PATH = '8.mp4' # Video dosyanızın yolu (Makaledeki gibi bir test videosu)
YOLO_MODEL = 'yolov8x.pt' # Makalede kullanılan model [1]
# COCO_FILE = 'coco.txt' # Supervision genellikle sınıf isimlerini modelden alır
FRAME_PROCESS_INTERVAL = 1 # Her karede işlem yap (Daha hassas zamanlama için, gerekirse artırılabilir)
TARGET_CLASSES_IDX = [1, 2, 3] # Takip edilecek COCO sınıf indeksleri (car: 2, bus: 5, truck: 7)

# --- Deneysel Hız Tahmini Ayarları (Makaledeki gibi) ---
# Bu değerler TAHMİNİDİR ve sahneye/videoya göre ayarlanması GEREKİR!
ASSUMED_CAR_WIDTH_METERS = 1.8 # Ortalama bir araba genişliği (metre) [6]

# Köşegen değişimine göre uzaklık ayarlama faktörleri (ÇOK DENEYSEL!) [4]
# Eşik değerleri (piksel/saniye cinsinden köşegen değişim hızı)
# Not: Orijinal kodunuzdaki eşikler negatif. Makaledeki mantık genellikle
# uzaklaşırken küçülme (negatif oran), yaklaşırken büyüme (pozitif oran) üzerinedir.
# Eşikleri ve faktörleri makaledeki mantığa ve kendi gözlemlerinize göre ayarlayın.
# Aşağıdaki değerler örnek amaçlıdır ve orijinal kodunuzdakilere benzerdir.
DIAG_RATE_THRESHOLD_NEAR = 50 # Hızlı büyüme eşiği (yakınlaşıyor olabilir) - Pozitif
DIAG_RATE_THRESHOLD_FAR = -10 # Yavaş küçülme/az büyüme eşiği (uzaklaşıyor olabilir) - Negatif/Küçük Pozitif
# Ayarlama Çarpanları
ADJUST_FACTOR_FAR = 1.8     # Uzaktaysa (yavaş değişiyorsa/küçülüyorsa) hızı artır
ADJUST_FACTOR_NEAR = 0.7    # Yakındaysa (hızlı büyüyorsa) hızı azalt
# -------------------------------------

# YOLO modelini yükle
try:
    model = YOLO(YOLO_MODEL)
except Exception as e:
    print(f"Hata: YOLO modeli yüklenemedi ({YOLO_MODEL}). Hata: {e}")
    exit()

# Video yakalamayı başlat
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Hata: Video dosyası açılamadı ({VIDEO_PATH}).")
    exit()

# Video bilgilerini al
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Uyarı: Video FPS değeri alınamadı. Varsayılan 30 kullanılıyor.")
    fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Kareler arası zaman (saniye cinsinden) - Her kare işleneceği için
frame_time_sec = 1 / fps

# ByteTrack takipçisini başlat [4]
# Supervision'ın ByteTrack'i genellikle varsayılan parametrelerle iyi çalışır
# track_thresh: Yüksek güvenli eşik, track_buffer: Kayıp kare toleransı, match_thresh: Eşleştirme eşiği
byte_tracker = sv.ByteTrack(frame_rate=fps, track_thresh=0.5, track_buffer=30, match_thresh=0.8)

# Yardımcı Supervision annotator'ları (isteğe bağlı, görselleştirme için)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)

# Araçların önceki PİKSEL konumlarını, zamanlarını ve KÖŞEGENLERİNİ saklamak için
# defaultdict kullanarak anahtar yoksa otomatik boş deque oluştur
prev_pixel_coords = defaultdict(lambda: deque(maxlen=2)) # Sadece son 2 konumu tut yeterli
prev_time_stamp = defaultdict(lambda: deque(maxlen=2)) # Sadece son 2 zamanı tut yeterli
prev_bbox_diag = defaultdict(lambda: deque(maxlen=2)) # Sadece son 2 köşegeni tut yeterli

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okuma hatası.")
        break

    frame_count += 1
    # FRAME_PROCESS_INTERVAL artık 1 olduğu için bu kontrol gereksiz, ama isterseniz ekleyebilirsiniz
    # if frame_count % FRAME_PROCESS_INTERVAL!= 0:
    #     continue

    # Zaman damgası
    current_time_sec = frame_count / fps

    # YOLOv8 ile nesne tespiti yap [4, 1]
    results = model(frame, verbose=False) # verbose=False sessiz mod

    # Supervision Detections formatına dönüştür
    detections = sv.Detections.from_ultralytics(results)

    # Sadece hedef sınıfları filtrele (car, truck, bus)
    detections = detections

    # ByteTrack ile takip et [4, 1]
    # Detections nesnesini tracker_id özelliği ile günceller
    detections = byte_tracker.update_with_detections(detections)

    # Hızları ve etiketleri hazırlamak için listeler
    labels =
    current_speeds_kmh = {} # Anlık hızları saklamak için

    # Takip edilen her araç için hız hesapla
    for detection_idx in range(len(detections)):
        # Supervision Detections'dan bilgileri al
        xyxy = detections.xyxy[detection_idx]
        confidence = detections.confidence[detection_idx]
        class_id = detections.class_id[detection_idx]
        tracker_id = detections.tracker_id[detection_idx]

        # Sınırlayıcı kutu koordinatları
        x1, y1, x2, y2 = int(xyxy), int(xyxy[4]), int(xyxy[1]), int(xyxy[5])

        # Referans noktası (alt-orta) - Piksel koordinatları (Makaledeki gibi)
        cx_pixel = (x1 + x2) // 2
        cy_pixel = y2
        current_pixel_coord = (cx_pixel, cy_pixel)

        # 1. Köşegen Uzaklığını Hesapla
        bbox_width_pixels = x2 - x1
        bbox_height_pixels = y2 - y1
        bbox_diag = 0
        if bbox_width_pixels > 0 and bbox_height_pixels > 0:
            bbox_diag = math.sqrt(bbox_width_pixels**2 + bbox_height_pixels**2)

        # Mevcut zamanı ve konumu kaydet
        prev_pixel_coords[tracker_id].append(current_pixel_coord)
        prev_time_stamp[tracker_id].append(current_time_sec)
        if bbox_diag > 0:
            prev_bbox_diag[tracker_id].append(bbox_diag)
        else:
            # Eğer köşegen 0 ise, önceki geçerli köşegeni tekrar ekle (oran hesaplaması için)
            if len(prev_bbox_diag[tracker_id]) > 0:
                prev_bbox_diag[tracker_id].append(prev_bbox_diag[tracker_id][-1])
            else:
                 prev_bbox_diag[tracker_id].append(0) # İlk kare ise 0 ekle


        speed_kmh = 0 # Varsayılan hız
        distance_adjustment_factor = 1.0 # Uzaklık ayar faktörü

        # Eğer bu araç için yeterli geçmiş veri varsa (en az 2 nokta) hızını hesapla
        if len(prev_pixel_coords[tracker_id]) >= 2 and len(prev_time_stamp[tracker_id]) >= 2:
            # Önceki ve mevcut koordinatları/zamanları al
            prev_coord_pix = prev_pixel_coords[tracker_id]
            curr_coord_pix = prev_pixel_coords[tracker_id][4]
            prev_time = prev_time_stamp[tracker_id]
            curr_time = prev_time_stamp[tracker_id][4]

            # Geçen süreyi hesapla (saniye)
            elapsed_time = curr_time - prev_time
            # VEYA sabit kare işleme aralığı kullan (eğer FRAME_PROCESS_INTERVAL > 1 ise):
            # elapsed_time = frame_time_sec

            if elapsed_time > 1e-6: # Çok küçük zaman farklarını veya sıfırı engelle
                # Piksel cinsinden mesafeyi hesapla [7]
                pixel_distance = math.dist(prev_coord_pix, curr_coord_pix)

                # Piksel cinsinden hızı hesapla (piksel/s) [4]
                speed_pixels_per_sec = pixel_distance / elapsed_time

                # 2. Gerçek Dünya Hızı Tahmini (Varsayılan Genişlik ile) [4, 6]
                scale_factor_m_per_pix = 0
                if bbox_width_pixels > 0:
                    # Makalede açı düzeltmesi belirtilmediği için kullanmıyoruz
                    scale_factor_m_per_pix = ASSUMED_CAR_WIDTH_METERS / bbox_width_pixels
                else:
                    scale_factor_m_per_pix = 0 # Hesaplama başarısız

                # --- İlk Hız Tahmini (m/s) ---
                speed_mps_initial = 0
                if scale_factor_m_per_pix > 0:
                    speed_mps_initial = speed_pixels_per_sec * scale_factor_m_per_pix

                # 3. Deneysel Uzaklık Ayarı (Köşegen Değişimi ile) [4]
                if len(prev_bbox_diag[tracker_id]) >= 2:
                    prev_diag = prev_bbox_diag[tracker_id]
                    curr_diag = prev_bbox_diag[tracker_id][4]

                    if prev_diag > 0 and curr_diag > 0: # Geçerli köşegen değerleri varsa
                        diag_change = curr_diag - prev_diag
                        diag_change_rate = diag_change / elapsed_time # piksel/saniye

                        # Eşiklere göre ayarlama faktörünü belirle
                        if diag_change_rate > DIAG_RATE_THRESHOLD_NEAR: # Hızlı büyüyor -> Yakın
                            distance_adjustment_factor = ADJUST_FACTOR_NEAR
                        elif diag_change_rate < DIAG_RATE_THRESHOLD_FAR: # Küçülüyor/Yavaş büyüyor -> Uzak
                            distance_adjustment_factor = ADJUST_FACTOR_FAR
                        # Diğer durumlar için (orta hızda büyüme) faktör 1.0 kalır

                        # print(f"ID: {tracker_id}, Diag Rate: {diag_change_rate:.1f}, Adj Factor: {distance_adjustment_factor}") # Debug
                    else:
                        # Geçerli köşegen yoksa ayarlama yapma
                        distance_adjustment_factor = 1.0
                else:
                     # Yeterli köşegen geçmişi yoksa ayarlama yapma
                     distance_adjustment_factor = 1.0

                # --- Nihai Hız Hesabı (km/h) ---
                speed_mps_final = speed_mps_initial * distance_adjustment_factor
                speed_kmh = speed_mps_final * 3.6

                # Mantıksız hızları filtrele (isteğe bağlı)
                if speed_kmh < 0: speed_kmh = 0
                if speed_kmh > 180: speed_kmh = 0 # Makul bir üst limit

                current_speeds_kmh[tracker_id] = speed_kmh # Hızı sakla

        # Görselleştirme için etiket oluştur
        label = f"ID:{tracker_id}"
        if tracker_id in current_speeds_kmh and current_speeds_kmh[tracker_id] > 0:
            label += f" {int(current_speeds_kmh[tracker_id])}km/h"
        # label += f" D:{int(bbox_diag)}" # İsteğe bağlı: Köşegeni göster
        # label += f" A:{distance_adjustment_factor:.1f}" # İsteğe bağlı: Ayar faktörünü göster
        labels.append(label)

        # Referans noktasını çiz (isteğe bağlı)
        cv2.circle(frame, (cx_pixel, cy_pixel), 3, (0, 255, 0), -1)

    # Supervision ile kare üzerine çizimleri yap
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # FPS bilgisini ekle (isteğe bağlı)
    processing_time = (time.time() - (frame_count / fps)) # Yaklaşık işlem süresi
    current_fps = 1.0 / (frame_time_sec + processing_time) if (frame_time_sec + processing_time) > 0 else fps
    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Hız Tahmini (YOLOv8 + ByteTrack + Deneysel Ayar)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC tuşuna basılınca çık
        break

cap.release()
cv2.destroyAllWindows()
