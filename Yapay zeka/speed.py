import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
# tracker.py dosyasının aynı dizinde olduğundan emin ol
from tracker import Tracker
import math
import time # Zaman ölçümü için eklendi

# --- Perspektif Dönüşümü için Ayarlar ---
# Bu noktaları kendi videonuza göre ayarlayın!
# 1. Video karesi üzerinde bilinen bir dikdörtgenin köşe piksellerini seçin
#    (örn. yol şeridinin köşeleri). Sıra önemli: Sol üst, Sağ üst, Sağ alt, Sol alt
pts_src = np.array([[250, 200], [700, 200], [950, 450], [50, 450]], dtype=np.float32) # Örnek piksel koordinatları

# 2. Bu piksellerin gerçek dünyadaki karşılıklarını metre cinsinden tanımlayın.
#    Örneğin, seçtiğiniz alan 5 metre genişliğinde ve 15 metre uzunluğunda olsun.
real_width_meters = 5.0
real_height_meters = 15.0
pts_dst = np.array([
    [0, 0],                                  # Sol üst
    [real_width_meters, 0],                  # Sağ üst
    [real_width_meters, real_height_meters], # Sağ alt
    [0, real_height_meters]                  # Sol alt
], dtype=np.float32)

# Perspektif dönüşüm matrisini hesapla
perspective_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
# -------------------------------------------

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture('8.mp4') # Video dosyasının adını kontrol et

# Video FPS'ini al
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Uyarı: Video FPS değeri alınamadı. Varsayılan 30 kullanılıyor.")
    fps = 30 # Veya bilinen bir değer girin

# Kareler arası zaman (saniye cinsinden)
# Her kareyi işlersek: frame_time_sec = 1.0 / fps
# Her 3 karede bir işlersek:
frame_process_interval = 3 # Her 3 karede bir işleme
frame_time_sec = frame_process_interval / fps

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()

# Araçların önceki konumlarını ve zamanlarını saklamak için
# Bu sefer gerçek dünya koordinatlarını saklayacağız
prev_real_coords = {}
prev_time_stamp = {}

# --- Kalibrasyon noktalarını çizmek için (opsiyonel) ---
def draw_calibration_points(frame, points):
    for i, pt in enumerate(points):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.polylines(frame, [points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
# ------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Zaman damgası (videonun başından itibaren geçen süre)
    # cap.get(cv2.CAP_PROP_POS_MSEC) her zaman güvenilir olmayabilir,
    # kare sayısına göre hesaplamak daha tutarlı olabilir.
    current_time_sec = count / fps

    count += 1
    if count % frame_process_interval != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    original_frame = frame.copy() # Orijinal kareyi sakla (çizimler için)

    # --- Kalibrasyon alanını çiz (opsiyonel) ---
    draw_calibration_points(frame, pts_src)
    # -------------------------------------------

    results = model.predict(frame, verbose=False) # verbose=False sessiz mod
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a.numpy()).astype("float")

    detections = []
    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c or 'truck' in c: # Kamyonları da ekleyebiliriz
             # Güven skorunu da kontrol etmek iyi olabilir:
             # conf = float(row[4])
             # if conf > 0.5: # Eşik değeri ayarlanabilir
             detections.append([x1, y1, x2, y2])

    bbox_id = tracker.update(detections)
    current_speeds = {} # Bu karedeki hızları saklamak için

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4) # Tam sayıya çevir

        # Aracın merkez noktasını bul (alt orta nokta genellikle daha stabildir)
        # cx = int(x3 + x4) // 2
        # cy = int(y3 + y4) // 2
        cx_pixel = int(x3 + x4) // 2
        cy_pixel = int(y4) # Alt orta nokta

        # Piksel koordinatını numpy formatına getir
        pixel_coord = np.array([[[cx_pixel, cy_pixel]]], dtype=np.float32)

        # Perspektif dönüşümü uygula
        # cv2.perspectiveTransform beklenen şekil (1, N, 2) olduğu için [[]] kullanıyoruz
        transformed_coord = cv2.perspectiveTransform(pixel_coord, perspective_matrix)

        # Dönüştürülmüş koordinatları al (artık metre cinsinden)
        # transformed_coord[0][0] -> [real_x, real_y]
        real_x = transformed_coord[0][0][0]
        real_y = transformed_coord[0][0][1]
        current_real_coord = (real_x, real_y)

        speed_kmh = 0 # Varsayılan hız

        # Eğer bu araç daha önce takip edildiyse hızını hesapla
        if id in prev_real_coords and id in prev_time_stamp:
            prev_coord = prev_real_coords[id]
            prev_time = prev_time_stamp[id]

            # Geçen süreyi hesapla (saniye)
            # elapsed_time = current_time_sec - prev_time
            # VEYA sabit kare işleme aralığı kullan:
            elapsed_time = frame_time_sec

            if elapsed_time > 0: # Zaman farkı sıfırdan büyükse
                # Gerçek dünyadaki mesafeyi hesapla (metre)
                distance_meters = math.dist(prev_coord, current_real_coord)

                # Hızı hesapla (m/s)
                speed_mps = distance_meters / elapsed_time

                # Hızı km/h'ye çevir
                speed_kmh = speed_mps * 3.6

                # Çok yüksek hızları filtrele (isteğe bağlı)
                if speed_kmh > 200: # Maksimum makul hız
                    speed_kmh = 0 # Hatalı ölçüm olabilir

                # print(f"ID {id}: Mesafe {distance_meters:.2f}m, Süre {elapsed_time:.3f}s, Hız {speed_kmh:.2f} km/h")

        # Mevcut durumu sonraki kare için sakla
        prev_real_coords[id] = current_real_coord
        prev_time_stamp[id] = current_time_sec # VEYA count / fps

        # --- Çizimler ---
        # Bounding box çiz
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        # ID'yi yazdır
        cv2.putText(frame, f"ID: {id}", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Hızı yazdır (hesaplanmışsa)
        if speed_kmh > 0:
            cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x4 - 50, y4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Merkez noktasını çiz (opsiyonel)
        cv2.circle(frame, (cx_pixel, cy_pixel), 3, (0, 255, 0), -1)


        # --- Orijinal isteğinizdeki Açı ve Köşegen Hesabı (Bilgi Amaçlı) ---
        # Bu hesaplamalar doğrudan güvenilir hız tahmini sağlamaz ama istenirse yapılabilir.
        bbox_width = x4 - x3
        bbox_height = y4 - y3
        bbox_diag = 0
        movement_angle_deg = 0

        # Önceki piksel koordinatları varsa açıyı hesapla
        # (Bu kısım için prev_cx, prev_cy gibi piksel tabanlı geçmiş de tutulmalı)
        # if id in prev_pixel_coords: # prev_pixel_coords diye bir dict tanımlamanız gerekir
        #     px, py = prev_pixel_coords[id]
        #     delta_x = cx_pixel - px
        #     delta_y = cy_pixel - py # Y ekseni aşağı doğru pozitif olduğundan işarete dikkat
        #     pixel_distance = math.sqrt(delta_x**2 + delta_y**2)
        #     if pixel_distance > 1: # Küçük hareketleri ihmal et
        #         movement_angle_rad = math.atan2(-delta_y, delta_x) # Y ekseni ters olduğu için -delta_y
        #         movement_angle_deg = math.degrees(movement_angle_rad)
        #         if movement_angle_deg < 0:
        #             movement_angle_deg += 360 # 0-360 derece arası

        # Köşegen
        if bbox_width > 0 and bbox_height > 0:
            bbox_diag = math.sqrt(bbox_width**2 + bbox_height**2)

        # Bu değerleri yazdırmak isterseniz:
        # cv2.putText(frame, f"Diag: {bbox_diag:.1f}", (x3, y3 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.putText(frame, f"Angle: {movement_angle_deg:.1f}", (x3, y3 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # --------------------------------------------------------------------


    cv2.imshow("Frame", frame)

    # ESC tuşuna basılınca çık
    if cv2.waitKey(1) & 0xFF == 27: # 1ms bekle, tuşa basılırsa kontrol et
        break

cap.release()
cv2.destroyAllWindows()
