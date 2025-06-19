# Gerekli kütüphaneleri içe aktarır:
import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from collections import defaultdict, deque

# --- Ayarlar Bölümü ---
VIDEO_PATH = 'pciletop.mkv'  # İşlenecek video dosyasının yolu.
YOLO_MODEL = 'yolov8x.pt' # Kullanılacak YOLOv8 modelinin dosya adı. Top için daha spesifik bir model gerekebilir.

# --- YENİ HFOV ve TOP AYARLARI ---
# Sağlanan bilgilere göre:
# Pet şişenin gerçek genişliği (Wgerc​ek​): 3 cm
# Pet şişenin görüntüdeki piksel genişliği (Ps​is​e1​): 316 piksel
# Görüntünün toplam yatay piksel genişliği (Ptoplam_yatay​): 1536 piksel
# Mesafe (D1): 31.32 cm
# HFOV: Yaklaşık 26.66 derece

TARGET_CLASS_NAME = 'sports ball' # Hedeflenecek sınıfın adı (YOLO modeline göre değişebilir)

BALL_DIAMETER_CM = 10.0 # Örnek bir top çapı (cm) - Kendi topunuzun çapını girin
HFOV_DEGREES = 64.05 # Kameranın yatay görüş açısı (derece)
# FRAME_WIDTH_PIXELS = 1536 # Bu, videodan otomatik alınacak (frame_width)

IOU_THRESHOLD = 0.3 # Intersection over Union (IoU) eşik değeri.
TRACK_EXPIRY_FRAMES = 15 # Bir nesnenin kaç kare boyunca tespit edilemezse takibinin sonlandırılacağını belirler.

# --- Etiket Güncelleme Ayarı ---
LABEL_UPDATE_INTERVAL_SEC = 0.5 # Etiketlerin ekranda güncellenme sıklığı (saniye).
# --------------------

try:
    model = YOLO(YOLO_MODEL)
    # Model sınıflarını alalım ve hedef sınıf ID'sini bulalım
    MODEL_CLASSES = model.names
    TARGET_CLASSES_IDX = [k for k, v in MODEL_CLASSES.items() if v.lower() == TARGET_CLASS_NAME.lower()]
    if not TARGET_CLASSES_IDX:
        print(f"Uyarı: Hedef sınıf '{TARGET_CLASS_NAME}' modelde bulunamadı. Lütfen sınıf adını veya ID'sini kontrol edin.")
        # Eğer sınıf adı bulunamazsa, kullanıcıya bilgi verilir.
        # Bu durumda, kod ilk güvenilir tespiti (confidence > 0.5) top olarak varsayacaktır.
        # En doğru sonuç için, topu güvenilir şekilde tespit eden bir model ve doğru sınıf adı/ID'si kullanılmalıdır.
except Exception as e:
    print(f"Hata: YOLO modeli yüklenemedi ({YOLO_MODEL}). Hata: {e}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Hata: Video dosyası açılamadı ({VIDEO_PATH}).")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
    print(f"Uyarı: FPS alınamadı, varsayılan {fps} kullanılıyor.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Bu bizim Ptoplam_yatay değerimiz olacak
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {VIDEO_PATH}, FPS: {fps:.2f}, Boyut: {frame_width}x{frame_height}")
if TARGET_CLASSES_IDX:
    print(f"Hedeflenen Sınıf ID(leri): {TARGET_CLASSES_IDX} ({TARGET_CLASS_NAME})")
else:
    print(f"Uyarı: '{TARGET_CLASS_NAME}' sınıfı modelde bulunamadığından, ilk güvenilir tespitler top olarak değerlendirilecektir.")
print(f"Top Çapı: {BALL_DIAMETER_CM} cm, HFOV: {HFOV_DEGREES} derece")


# Piksel başına düşen açısal çözünürlük (derece/piksel)
# Ac¸​ı / Piksel = HFOV / Ptoplam_yatay​ (Düzeltme: HFOV / Ptoplam_yatay olmalı)
if frame_width > 0 :
    angular_resolution_deg_per_pixel = HFOV_DEGREES / frame_width
    print(f"Açısal Çözünürlük: {angular_resolution_deg_per_pixel:.5f} derece/piksel")
else:
    print("Hata: Frame genişliği 0, açısal çözünürlük hesaplanamıyor.")
    exit()


next_track_id = 0
prev_tracks = {}
track_history_coords = defaultdict(lambda: deque(maxlen=10)) # XY koordinatları için geçmiş (hız için en az 2 gerekir), iz yolu için artırıldı
track_history_times = defaultdict(lambda: deque(maxlen=5))  # Zaman damgaları
track_history_distances = defaultdict(lambda: deque(maxlen=5)) # Hesaplanan mesafeler (cm)

track_last_label_update_time = defaultdict(float)
track_cached_label_parts = defaultdict(list)

frame_count_processed = 0 # Sadece işlenen kareleri saymak için
processing_start_time = time.time()

def calculate_iou(box1, box2):
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    inter_width = max(0, x2_i - x1_i)
    inter_height = max(0, y2_i - y1_i)
    inter_area = inter_width * inter_height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- YENİ: Mesafe Hesaplama Fonksiyonu ---
def calculate_distance_cm(object_pixel_width, actual_object_width_cm, global_angular_resolution_deg_per_pixel):
    if object_pixel_width <= 0 or actual_object_width_cm <=0 or global_angular_resolution_deg_per_pixel <=0:
        return None

    # 1. Nesnenin Kapladığı Toplam Açısal Genişlik (theta_object)
    theta_object_degrees = object_pixel_width * global_angular_resolution_deg_per_pixel

    # 2. Mesafe Hesaplama
    # D = W_gerçek / (2 * tan(theta_object_degrees / 2))
    # tan fonksiyonu radyan cinsinden açı bekler
    if theta_object_degrees <= 0 or theta_object_degrees >= 180: # Açı geçersizse
        return None

    try:
        # Açının yarısını radyana çevir
        angle_rad_half = math.radians(theta_object_degrees / 2.0)
        if math.tan(angle_rad_half) == 0: # tan(0) durumunu engelle
             return None
        distance = actual_object_width_cm / (2.0 * math.tan(angle_rad_half))
    except (ZeroDivisionError, ValueError):
        return None # Bölme hatası veya tanım dışı durum
    return distance

# Video karelerini işleme döngüsü
actual_frame_num = 0
while True:
    ret, frame = cap.read()
    actual_frame_num += 1
    if not ret:
        print("Video bitti veya okuma hatası.")
        break

    frame_count_processed += 1
    current_time_sec = actual_frame_num / fps # Orijinal video zaman damgası

    results = model(frame, verbose=False, classes=TARGET_CLASSES_IDX if TARGET_CLASSES_IDX else None)[0] # Sınıf filtresini burada uygula
    current_detections_raw = []

    if results.boxes is not None:
        for i in range(len(results.boxes.xyxy)):
            xyxy = results.boxes.xyxy[i].tolist()
            confidence = results.boxes.conf[i].item()
            class_id = int(results.boxes.cls[i].item()) # Tespit edilen sınıf ID'si

            # Eğer TARGET_CLASSES_IDX tanımlıysa ve ID eşleşiyorsa VEYA
            # TARGET_CLASSES_IDX tanımlı değilse (sınıf bulunamadıysa) ve güven skoru yüksekse al.
            if (TARGET_CLASSES_IDX and class_id in TARGET_CLASSES_IDX) or \
               (not TARGET_CLASSES_IDX and confidence > 0.5): # Güven eşiği 0.5 olarak ayarlandı
                current_detections_raw.append({
                    'bbox': xyxy, 'confidence': confidence, 'class_id': class_id,
                    'matched_track_id': None
                })

    matched_current_indices = set()
    updated_prev_tracks = {}

    for track_id, track_info in prev_tracks.items():
        best_match_iou = 0
        best_match_idx = -1
        for i, current_det in enumerate(current_detections_raw):
            if i in matched_current_indices:
                continue
            iou = calculate_iou(track_info['bbox'], current_det['bbox'])
            if iou > best_match_iou:
                best_match_iou = iou
                best_match_idx = i

        if best_match_iou >= IOU_THRESHOLD and best_match_idx != -1:
            current_detections_raw[best_match_idx]['matched_track_id'] = track_id
            matched_current_indices.add(best_match_idx)
            
            updated_prev_tracks[track_id] = {
                'bbox': current_detections_raw[best_match_idx]['bbox'],
                'timestamp': current_time_sec,
                'frames_since_last_seen': 0,
                'class_id': current_detections_raw[best_match_idx]['class_id']
            }
        else:
            track_info['frames_since_last_seen'] += 1
            if track_info['frames_since_last_seen'] < TRACK_EXPIRY_FRAMES:
                updated_prev_tracks[track_id] = track_info

    for i, current_det in enumerate(current_detections_raw):
        if i not in matched_current_indices:
            new_track_id = next_track_id
            next_track_id += 1
            current_det['matched_track_id'] = new_track_id
            updated_prev_tracks[new_track_id] = {
                'bbox': current_det['bbox'],
                'timestamp': current_time_sec,
                'frames_since_last_seen': 0,
                'class_id': current_det['class_id']
            }

    prev_tracks = updated_prev_tracks
    annotated_frame = frame.copy()

    for track_id, track_info in prev_tracks.items():
        if track_info['frames_since_last_seen'] > 0:
            continue

        xyxy = track_info['bbox']
        current_obj_time = track_info['timestamp']
        x1, y1, x2, y2 = map(int, xyxy)
        
        current_ball_pixel_width = float(x2 - x1)
        cx_pixel = (x1 + x2) // 2
        cy_pixel = (y1 + y2) // 2 
        current_pixel_coord = (cx_pixel, cy_pixel)

        current_distance_cm = calculate_distance_cm(
            current_ball_pixel_width,
            BALL_DIAMETER_CM,
            angular_resolution_deg_per_pixel # Global değişkeni kullan
        )

        track_history_coords[track_id].append(current_pixel_coord)
        track_history_times[track_id].append(current_obj_time)
        if current_distance_cm is not None and current_distance_cm > 0: # Sadece pozitif mesafeleri kaydet
            track_history_distances[track_id].append(current_distance_cm)

        speed_mps_radial = 0.0
        speed_mps_xy = 0.0
        distance_str = "Mesafe: N/A"
        speed_str = "Hız: N/A"

        if current_distance_cm is not None and current_distance_cm > 0:
            distance_str = f"Mesafe: {current_distance_cm:.1f} cm"

        if len(track_history_times[track_id]) >= 2 and len(track_history_coords[track_id]) >=2:
            prev_time_idx = -2 
            curr_time_idx = -1
            
            # Eğer yeterli mesafe verisi yoksa, daha eski zaman damgalarını kullanmayı dene (hız için en az 2 mesafe noktası gerekir)
            if len(track_history_distances[track_id]) < 2 and len(track_history_times[track_id]) > 2:
                 # Zaman ve koordinat geçmişinden sondan 3. ve 2. elemanları almayı deneyebiliriz,
                 # ancak bu, mesafe verisinin olmaması durumunda tutarsızlığa yol açabilir.
                 # Şimdilik, mesafe geçmişi 2'den azsa radyal hızı hesaplama.
                 pass


            prev_time = track_history_times[track_id][prev_time_idx]
            curr_time = track_history_times[track_id][curr_time_idx]
            elapsed_time = curr_time - prev_time

            if elapsed_time > 0.001: 
                # 1. Radyal Hız (Mesafe Değişiminden)
                if len(track_history_distances[track_id]) >= 2:
                    prev_dist_cm = track_history_distances[track_id][-2] # Sondan ikinci mesafe
                    curr_dist_cm = track_history_distances[track_id][-1] # Son mesafe
                    
                    # Zaman damgalarının bu mesafelerle senkronize olduğundan emin olmalıyız.
                    # Basitlik adına, son iki mesafe ve son iki zaman damgasını kullanıyoruz.
                    # Daha kesin olmak için, her mesafenin kaydedildiği zamanı ayrıca saklamak gerekebilir.

                    distance_change_cm = curr_dist_cm - prev_dist_cm 
                    speed_mps_radial = (distance_change_cm / 100.0) / elapsed_time 

                # 2. XY Düzlemindeki Hız (Piksel Değişiminden)
                prev_coord_pix = track_history_coords[track_id][-2]
                curr_coord_pix = track_history_coords[track_id][-1]
                
                m_per_pixel_at_object = 0
                # XY hızı için ortalama veya son mesafeyi kullan
                if track_history_distances[track_id]: # En az bir mesafe kaydı varsa
                    avg_dist_for_xy_m = track_history_distances[track_id][-1] / 100.0 # Son mesafeyi m cinsinden al
                    if current_ball_pixel_width > 0 and BALL_DIAMETER_CM > 0:
                         m_per_pixel_at_object = (BALL_DIAMETER_CM / 100.0) / current_ball_pixel_width
                
                if m_per_pixel_at_object > 0:
                    pixel_distance_moved = math.dist(prev_coord_pix, curr_coord_pix)
                    real_world_distance_moved_xy_m = pixel_distance_moved * m_per_pixel_at_object
                    speed_mps_xy = real_world_distance_moved_xy_m / elapsed_time
                
                if (current_distance_cm is not None and current_distance_cm > 0):
                    direction = ""
                    if abs(speed_mps_radial) > 0.01: 
                        if speed_mps_radial > 0.01 : direction = "Uzaklaşıyor"
                        elif speed_mps_radial < -0.01 : direction = "Yaklaşıyor"
                        speed_str = f"Hız(rad): {abs(speed_mps_radial):.2f} m/s {direction}"
                        if abs(speed_mps_xy) > 0.01 : # Eğer XY hızı da anlamlıysa ekle
                             speed_str += f" | XY: {speed_mps_xy:.2f} m/s"
                    elif abs(speed_mps_xy) > 0.01 : # Sadece XY hızı anlamlıysa
                         speed_str = f"Hız(xy): {speed_mps_xy:.2f} m/s"
                    else:
                         speed_str = "Hız: <0.01 m/s"


        if not track_cached_label_parts[track_id] or \
           (current_time_sec - track_last_label_update_time[track_id] >= LABEL_UPDATE_INTERVAL_SEC):
            
            current_label_parts = [f"ID:{track_id}"]
            current_label_parts.append(distance_str)
            if speed_str != "Hız: N/A": 
                 current_label_parts.append(speed_str)

            track_cached_label_parts[track_id] = current_label_parts
            track_last_label_update_time[track_id] = current_time_sec

        box_color = (0, 255, 0) 
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        
        if len(track_history_coords[track_id]) > 1:
            points = np.array(list(track_history_coords[track_id]), dtype=np.int32)
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2) # Kırmızı iz

        label_y_start_pos = y1 - 7
        if y1 < 30: label_y_start_pos = y2 + 20 # Eğer kutu ekranın tepesine çok yakınsa etiketi altına al

        if track_cached_label_parts[track_id]:
            for i, part_text in enumerate(track_cached_label_parts[track_id]):
                text_y_pos = label_y_start_pos + (i * 18) if y1 < 30 else label_y_start_pos - (i * 18)
                cv2.putText(annotated_frame, part_text, (x1 + 3, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA) 
                cv2.putText(annotated_frame, part_text, (x1 + 3, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA) 

    ids_to_delete_from_history = [tid for tid in track_history_coords.keys() if tid not in prev_tracks]
    for tid_del in ids_to_delete_from_history:
        if tid_del in track_history_coords: del track_history_coords[tid_del]
        if tid_del in track_history_times: del track_history_times[tid_del]
        if tid_del in track_history_distances: del track_history_distances[tid_del]
        if tid_del in track_last_label_update_time: del track_last_label_update_time[tid_del]
        if tid_del in track_cached_label_parts: del track_cached_label_parts[tid_del]

    elapsed_since_start = time.time() - processing_start_time
    actual_processing_fps = frame_count_processed / elapsed_since_start if elapsed_since_start > 0 else 0
    cv2.putText(annotated_frame, f"Processing FPS: {actual_processing_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"HFOV: {HFOV_DEGREES:.2f}deg, BallDia: {BALL_DIAMETER_CM:.1f}cm", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.imshow("Top Mesafe ve Hiz Tespiti (HFOV Metodu)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC tuşu
        break

cap.release()
cv2.destroyAllWindows()
print("İşlem tamamlandı.")

