# Gerekli kütüphaneleri içe aktarır:
import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from collections import defaultdict, deque
import os # Resim dosyalarını listelemek için eklendi

# --- Ayarlar Bölümü ---
IMAGE_PATH_OR_FOLDER = 'topresim1.jpeg' # İşlenecek resim dosyasının veya klasörünün yolu.
                                            # Eğer bir klasör ise, içindeki tüm .jpg, .png, .jpeg dosyaları işlenir.
YOLO_MODEL = 'yolov8x.pt' # Kullanılacak YOLOv8 modelinin dosya adı.

TARGET_CLASS_NAME = 'sports ball' # Hedeflenecek sınıfın adı (YOLO modeline göre değişebilir)

BALL_DIAMETER_CM = 9.0 # Örnek bir top çapı (cm) - Kendi topunuzun çapını girin
HFOV_DEGREES = 98.72 # Kameranın yatay görüş açısı (derece)

IOU_THRESHOLD = 0.3 # Intersection over Union (IoU) eşik değeri.
TRACK_EXPIRY_FRAMES = 15 # Tek resim için bu ayarın doğrudan etkisi olmayacak.
LABEL_UPDATE_INTERVAL_SEC = 0.5 # Tek resim için bu ayarın doğrudan etkisi olmayacak.
# --------------------
try:
    model = YOLO(YOLO_MODEL)
    # print(model) # Bu satır çok fazla çıktı verebilir, şimdilik kapalı kalsın.
    MODEL_CLASSES = model.names
    print(f"MODEL TÜM SINIFLARI: {MODEL_CLASSES}") # Modelin tüm sınıflarını görmek için eklendi
    TARGET_CLASSES_IDX = [k for k, v in MODEL_CLASSES.items() if v.lower() == TARGET_CLASS_NAME.lower()]
    print(f"HEDEFLENEN SINIF ADI: '{TARGET_CLASS_NAME}'") # Ayarladığınız sınıf adını teyit edin
    print(f"BULUNAN HEDEF SINIF ID(LERİ): {TARGET_CLASSES_IDX}") # ID'nin bulunup bulunmadığını kontrol edin

    if not TARGET_CLASSES_IDX:
        print(f"UYARI: Hedef sınıf '{TARGET_CLASS_NAME}' modelde bulunamadı. Lütfen sınıf adını veya ID'sini kontrol edin.")
except Exception as e:
    print(f"Hata: YOLO modeli yüklenemedi ({YOLO_MODEL}). Hata: {e}")
    exit()

def get_image_files(path):
    """Belirtilen yoldaki resim dosyalarını veya tek bir resim dosyasını döndürür."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in image_extensions]
    elif os.path.isfile(path) and os.path.splitext(path)[1].lower() in image_extensions:
        return [path]
    else:
        print(f"Hata: Geçerli bir resim dosyası veya klasörü bulunamadı: {path}")
        return []

image_files = get_image_files(IMAGE_PATH_OR_FOLDER)
if not image_files:
    exit()

# Piksel başına düşen açısal çözünürlük (derece/piksel)
# Bu, her resim için ayrıca hesaplanacak.
angular_resolution_deg_per_pixel = 0

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

def calculate_distance_cm(object_pixel_width, actual_object_width_cm, global_angular_resolution_deg_per_pixel):
    if object_pixel_width <= 0 or actual_object_width_cm <=0 or global_angular_resolution_deg_per_pixel <=0:
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

# Her resim için bu döngü çalışacak
for image_path_current in image_files:
    frame = cv2.imread(image_path_current)
    if frame is None:
        print(f"Hata: Resim dosyası okunamadı ({image_path_current}).")
        continue

    processing_start_time = time.time()

    frame_height, frame_width = frame.shape[:2]
    print(f"Resim: {image_path_current}, Boyut: {frame_width}x{frame_height}")

    if frame_width > 0:
        angular_resolution_deg_per_pixel = HFOV_DEGREES / frame_width
        print(f"Açısal Çözünürlük: {angular_resolution_deg_per_pixel:.5f} derece/piksel")
    else:
        print("Hata: Frame genişliği 0, açısal çözünürlük hesaplanamıyor.")
        continue

    if TARGET_CLASSES_IDX:
        print(f"Hedeflenen Sınıf ID(leri): {TARGET_CLASSES_IDX} ({TARGET_CLASS_NAME})")
    else:
        print(f"Uyarı: '{TARGET_CLASS_NAME}' sınıfı modelde bulunamadığından, ilk güvenilir tespitler top olarak değerlendirilecektir.")
    print(f"Top Çapı: {BALL_DIAMETER_CM} cm, HFOV: {HFOV_DEGREES} derece")

    # --- Tek resim için izleme değişkenleri sıfırlanır ---
    next_track_id = 0
    # prev_tracks bir önceki "kare"den (ki burada yok) gelen izleri tutar.
    # Tek resim için her zaman boş başlayacak.
    prev_tracks = {}
    track_history_coords = defaultdict(lambda: deque(maxlen=10))
    # track_history_times ve distances hız hesaplaması için, tek resimde N/A olacak
    track_history_times = defaultdict(lambda: deque(maxlen=5))
    track_history_distances = defaultdict(lambda: deque(maxlen=5))
    track_last_label_update_time = defaultdict(float) # Tek resim için çok anlamlı değil
    track_cached_label_parts = defaultdict(list)
    # ----------------------------------------------------

    # Tek bir "kare" olduğu için zaman damgası sabit olabilir.
    # Hız hesaplamaları zaten en az iki zaman damgası gerektirdiğinden etkilenmeyecek.
    current_time_sec = 0.0 # Sabit zaman

    results = model(frame, verbose=False, classes=TARGET_CLASSES_IDX if TARGET_CLASSES_IDX else None)[0]
    current_detections_raw = []

    if results.boxes is not None:
        for i in range(len(results.boxes.xyxy)):
            xyxy = results.boxes.xyxy[i].tolist()
            confidence = results.boxes.conf[i].item()
            class_id = int(results.boxes.cls[i].item())

            if (TARGET_CLASSES_IDX and class_id in TARGET_CLASSES_IDX) or \
               (not TARGET_CLASSES_IDX and confidence > 0.01):
                current_detections_raw.append({
                    'bbox': xyxy, 'confidence': confidence, 'class_id': class_id,
                    'matched_track_id': None
                })

    # --- İzleme Mantığı (Tek Resim için Basitleştirilmiş) ---
    # Bu kısım, videodaki gibi çalışacak ancak prev_tracks her zaman boş olduğu için
    # tüm tespitler yeni track olarak atanacak.
    matched_current_indices = set()
    updated_prev_tracks = {} # Bu, mevcut resimdeki izleri tutacak

    # prev_tracks boş olduğu için bu döngü ilk başta çalışmayacak.
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
            # Bu kısım tek resimde pek çalışmaz, çünkü prev_tracks hep boş.
            pass # Normalde burada eşleşen iz güncellenirdi.
        else:
            # Bu kısım da pek çalışmaz.
            pass

    # Tüm tespitler yeni izler olarak kabul edilir.
    for i, current_det in enumerate(current_detections_raw):
        # if i not in matched_current_indices: # Bu kontrol gereksiz çünkü matched_current_indices hep boş kalacak
        new_track_id = next_track_id
        next_track_id += 1
        current_det['matched_track_id'] = new_track_id
        updated_prev_tracks[new_track_id] = {
            'bbox': current_det['bbox'],
            'timestamp': current_time_sec, # Sabit zaman
            'frames_since_last_seen': 0, # Her zaman 0
            'class_id': current_det['class_id']
        }

    # prev_tracks'i mevcut resmin izleriyle güncelliyoruz (bir sonraki "sanal" kare için, ki olmayacak)
    # Ancak çizim için updated_prev_tracks kullanılacak.
    # Aslında, `prev_tracks` yerine doğrudan `updated_prev_tracks`'i çizim için kullanabiliriz.
    # Ya da kafa karışıklığını önlemek için `current_tracks_in_frame` gibi bir isim verebiliriz.
    # Orijinal yapıya sadık kalmak adına `prev_tracks`'i güncelleyip onu kullanalım.
    prev_tracks = updated_prev_tracks
    # --- İzleme Mantığı Sonu ---

    annotated_frame = frame.copy()

    for track_id, track_info in prev_tracks.items(): # prev_tracks artık mevcut resmin tespitlerini içeriyor
        # frames_since_last_seen her zaman 0 olacağı için bu koşul gereksiz
        # if track_info['frames_since_last_seen'] > 0:
        #     continue

        xyxy = track_info['bbox']
        current_obj_time = track_info['timestamp'] # Bu 0.0 olacak
        x1, y1, x2, y2 = map(int, xyxy)

        current_ball_pixel_width = float(x2 - x1)
        cx_pixel = (x1 + x2) // 2
        cy_pixel = (y1 + y2) // 2
        current_pixel_coord = (cx_pixel, cy_pixel)

        current_distance_cm = calculate_distance_cm(
            current_ball_pixel_width,
            BALL_DIAMETER_CM,
            angular_resolution_deg_per_pixel
        )

        track_history_coords[track_id].append(current_pixel_coord) # Tek nokta eklenecek
        track_history_times[track_id].append(current_obj_time)     # Tek zaman (0.0) eklenecek
        if current_distance_cm is not None and current_distance_cm > 0:
            track_history_distances[track_id].append(current_distance_cm) # Tek mesafe eklenecek

        # Hız hesaplamaları tek resim için "N/A" olacaktır
        speed_mps_radial = 0.0
        speed_mps_xy = 0.0
        distance_str = "Mesafe: N/A"
        speed_str = "Hız: N/A" # Varsayılan olarak N/A

        if current_distance_cm is not None and current_distance_cm > 0:
            distance_str = f"Mesafe: {current_distance_cm:.1f} cm"

        # Hız için en az 2 kayıt gerekir, tek resimde bu koşul sağlanmayacak
        if len(track_history_times[track_id]) >= 2 and len(track_history_coords[track_id]) >=2:
            # Bu blok tek resim için çalışmayacak, speed_str "N/A" kalacak
            # Orijinal hız hesaplama mantığı burada yer alıyordu,
            # ancak tek resimde anlamlı olmadığı için varsayılan değerler kullanılacak.
            pass # Hız hesaplama mantığı buraya kopyalanabilir ama sonuç N/A olacaktır.

        # Etiketleme mantığı
        # if not track_cached_label_parts[track_id] or \
        #    (current_time_sec - track_last_label_update_time[track_id] >= LABEL_UPDATE_INTERVAL_SEC):
        # Yukarıdaki zaman bazlı etiket güncelleme tek resimde gereksiz, her zaman güncel olacak.

        current_label_parts = [f"ID:{track_id}"]
        current_label_parts.append(distance_str)
        # speed_str zaten "Hız: N/A" olarak ayarlandı, o yüzden eklenmesine gerek yok
        # if speed_str != "Hız: N/A": # Bu koşul her zaman false olacak
        current_label_parts.append(speed_str) # "Hız: N/A" eklenecek

        track_cached_label_parts[track_id] = current_label_parts
        # track_last_label_update_time[track_id] = current_time_sec # Çok gerekli değil

        box_color = (0, 255, 0)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

        # İz yolu çizimi (tek nokta olduğu için çizilmeyecek)
        if len(track_history_coords[track_id]) > 1: # Bu koşul sağlanmayacak
            points = np.array(list(track_history_coords[track_id]), dtype=np.int32)
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)

        label_y_start_pos = y1 - 7
        if y1 < 30: label_y_start_pos = y2 + 20

        if track_cached_label_parts[track_id]:
            for i, part_text in enumerate(track_cached_label_parts[track_id]):
                text_y_pos = label_y_start_pos + (i * 18) # Etiketleri kutunun üstüne sırala
                if y1 < (len(track_cached_label_parts[track_id]) * 18) + 7 : # Eğer etiketler yukarı taşarsa
                    text_y_pos = y2 + 20 + (i * 18) # Etiketleri kutunun altına al
                
                cv2.putText(annotated_frame, part_text, (x1 + 3, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, part_text, (x1 + 3, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

    # Geçmişi temizleme (tek resim için çok etkisi yok)
    # ids_to_delete_from_history = [tid for tid in track_history_coords.keys() if tid not in prev_tracks]
    # for tid_del in ids_to_delete_from_history:
    #     if tid_del in track_history_coords: del track_history_coords[tid_del]
    #     # ... diğer geçmişler için de aynı

    elapsed_since_start = time.time() - processing_start_time
    processing_time_ms = elapsed_since_start * 1000
    cv2.putText(annotated_frame, f"Processing Time: {processing_time_ms:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"HFOV: {HFOV_DEGREES:.2f}deg, BallDia: {BALL_DIAMETER_CM:.1f}cm", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.imshow(f"Top Mesafe Tespiti - {os.path.basename(image_path_current)}", annotated_frame)
    print(f"'{os.path.basename(image_path_current)}' için işlem tamamlandı. Kapatmak için bir tuşa basın.")
    cv2.waitKey(0) # Bir tuşa basılana kadar bekle
    cv2.destroyWindow(f"Top Mesafe Tespiti - {os.path.basename(image_path_current)}") # Sadece aktif pencereyi kapat

# Tüm pencereleri kapat (eğer döngüden sonra hala açık kalan varsa)
# cv2.destroyAllWindows() # Bu, tüm resimler işlendikten sonra en sonda olabilir veya her pencere kendi kendine kapanır.
print("Tüm resimlerin işlenmesi tamamlandı.")
