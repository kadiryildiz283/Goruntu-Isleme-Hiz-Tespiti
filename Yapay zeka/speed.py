import cv2
# import pandas as pd # pandas bu kodda kullanılmıyor
import numpy as np
from ultralytics import YOLO
# import supervision as sv # Supervision kaldırılıyor
import math
import time
from collections import defaultdict, deque # Önceki koordinatları saklamak için

# --- Ayarlar ---
VIDEO_PATH = '8.mp4' # Video dosyanızın yolu
YOLO_MODEL = 'yolov8x.pt' # Kullanılacak YOLO modeli
# COCO_FILE = 'coco.txt' # Artık gerek yok
# FRAME_PROCESS_INTERVAL = 1 # Her karede işlem yapılacak, sabit kalabilir
TARGET_CLASSES_IDX = [2, 5, 7] # Takip edilecek COCO sınıf indeksleri (car: 2, bus: 5, truck: 7) - Not: Orijinalde 1,2,3 vardı, araçlar için 2,5,7 daha uygun.

# --- Basit IoU Takip Ayarları (Supervision/ByteTrack yerine) ---
IOU_THRESHOLD = 0.5 # İki sınırlayıcı kutuyu aynı nesneye ait saymak için IoU eşiği
TRACK_EXPIRY_FRAMES = 15 # Bir nesne kaç kare boyunca algılanmazsa takibi bırakılacak

# --- Deneysel Hız Tahmini Ayarları ---
# Bu değerler VİDEOYA ÖZELDİR ve sahnenize/kameranıza göre AYARLANMALIDIR!
ASSUMED_CAR_WIDTH_METERS = 1.8 # Hız hesaplaması için varsayılan araç genişliği (metre)

# Köşegen değişimine göre uzaklık ayarlama faktörleri (ÇOK DENEYSEL!)
# Bu eşikler ve faktörler, araç yakınlaşırken/uzaklaşırken hız tahminini
# kalibre etmek için kullanılır. Deneysel olarak ayarlanmaları gerekir.
DIAG_RATE_THRESHOLD_NEAR = 40 # Köşegen büyüme hızı (piksel/s) - Yakınlaşma eşiği
DIAG_RATE_THRESHOLD_FAR = -15 # Köşegen küçülme hızı (piksel/s) - Uzaklaşma eşiği
ADJUST_FACTOR_FAR = 1.7     # Uzaklaşan araçlar için hızı artırma çarpanı
ADJUST_FACTOR_NEAR = 0.8    # Yakınlaşan araçlar için hızı azaltma çarpanı
# -------------------------------------

# YOLO modelini yükle
try:
    model = YOLO(YOLO_MODEL)
    # Modeli kullanmadan önce sınıf isimlerini kontrol etmek faydalı olabilir:
    # print(model.names)
except Exception as e:
    print(f"Hata: YOLO modeli yüklenemedi ({YOLO_MODEL}). Lütfen dosya yolunu ve internet bağlantısını kontrol edin. Hata: {e}")
    exit()

# Video yakalamayı başlat
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Hata: Video dosyası açılamadı ({VIDEO_PATH}). Lütfen dosya yolunu kontrol edin.")
    exit()

# Video bilgilerini al
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Uyarı: Video FPS değeri alınamadı veya sıfır. Varsayılan 30 kullanılıyor.")
    fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {VIDEO_PATH}, FPS: {fps}, Boyut: {frame_width}x{frame_height}")

# --- Takip ve Hız İçin Durum Değişkenleri ---
next_track_id = 0 # Yeni takip kimlikleri atamak için sayaç
# Önceki karedeki takip edilen nesnelerin bilgileri (Track ID -> Bilgiler)
# IoU takibi için kullanılacak ana yapı
prev_tracks = {} # {track_id: {'bbox': [x1, y1, x2, y2], 'timestamp': t, 'diag': diag_pix, 'frames_since_last_seen': n, 'class_id': cls}}

# Araçların önceki PİKSEL konumlarını (alt-orta), zamanlarını ve KÖŞEGENLERİNİ saklamak için
# deque(maxlen=2): Sadece en son 2 değeri tutar (önceki ve mevcut)
# Bu yapı, track_id'lere göre hız hesaplama geçmişini tutacak.
track_history_coords = defaultdict(lambda: deque(maxlen=2))
track_history_times = defaultdict(lambda: deque(maxlen=2))
track_history_diags = defaultdict(lambda: deque(maxlen=2))

frame_count = 0
processing_start_time = time.time() # Gerçek işleme süresini ölçmek için

# IoU hesaplama fonksiyonu
def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
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

    if union_area == 0:
        return 0
    return inter_area / union_area

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okuma hatası.")
        break

    frame_count += 1
    # Video zamanına göre hız hesaplaması için:
    current_time_sec = frame_count / fps
    # Gerçek zamana göre (işlem süresini içerir, FPS dalgalanırsa hız tahmini etkilenebilir)
    # current_time_sec = time.time()

    # YOLOv8 ile nesne tespiti yap
    # YOLOv8 model() metodu bir liste döndürür, ilk elementi alıyoruz
    results = model(frame, verbose=False)[0]

    # Tespit edilen nesnelerin bilgileri
    current_detections_raw = []
    if results.boxes is not None:
        # results.boxes.xyxy, results.boxes.conf, results.boxes.cls numpy dizileridir
        for i in range(len(results.boxes.xyxy)):
            xyxy = results.boxes.xyxy[i].tolist()
            confidence = results.boxes.conf[i].item()
            class_id = int(results.boxes.cls[i].item())

            # Sadece hedef sınıfları dahil et
            if class_id in TARGET_CLASSES_IDX: # TARGET_CLASSES_IDX boş değilse kontrol eder
                 current_detections_raw.append({'bbox': xyxy, 'confidence': confidence, 'class_id': class_id, 'matched_track_id': None})


    # --- Basit IoU Takip Mantığı ---
    # Yeni karedeki tespitleri (current_detections_raw), önceki karedeki takip edilen nesnelerle (prev_tracks) eşleştir
    matched_current_indices = set() # Bu karede eşleşen tespitlerin indexleri
    updated_prev_tracks = {} # Bir sonraki kare için güncellenmiş takip listesi

    # Önceki takip edilen nesnelerle mevcut tespitleri karşılaştır (Eşleştirme)
    for track_id, track_info in prev_tracks.items():
        best_match_iou = 0
        best_match_idx = -1

        # Mevcut tespitler arasında bu takip edilen nesneye en çok benzeyeni bul (En yüksek IoU)
        for i, current_det in enumerate(current_detections_raw):
            if i in matched_current_indices: # Bu tespit zaten başka bir takiple eşleştiyse atla (bir tespit sadece bir takiple eşleşebilir)
                continue

            iou = calculate_iou(track_info['bbox'], current_det['bbox'])

            if iou > best_match_iou:
                best_match_iou = iou
                best_match_idx = i

        # Eğer yeterli IoU ile bir eşleşme bulunduysa (bir önceki nesne bulunduysa)
        if best_match_iou >= IOU_THRESHOLD and best_match_idx != -1:
            # Eşleşen mevcut tespiti bu takip kimliğine ata
            current_detections_raw[best_match_idx]['matched_track_id'] = track_id
            matched_current_indices.add(best_match_idx) # Bu tespiti eşleşti olarak işaretle

            # Takip bilgisini güncelle (Konum, zaman, diag güncellendi, görülme sayısı sıfırlandı)
            x1, y1, x2, y2 = current_detections_raw[best_match_idx]['bbox']
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_diag = math.sqrt(bbox_width**2 + bbox_height**2) if bbox_width > 0 and bbox_height > 0 else 0

            updated_prev_tracks[track_id] = {
                'bbox': current_detections_raw[best_match_idx]['bbox'],
                'timestamp': current_time_sec,
                'diag': bbox_diag if bbox_diag > 0 else track_info.get('diag', 0), # Geçerli diag yoksa önceki geçerliyi koru
                'frames_since_last_seen': 0, # Yeni görüldü
                'class_id': current_detections_raw[best_match_idx]['class_id'] # Sınıf bilgisini de tut
            }
        else:
            # Bu takip edilen nesne mevcut karede eşleşmedi
            track_info['frames_since_last_seen'] += 1
            if track_info['frames_since_last_seen'] < TRACK_EXPIRY_FRAMES:
                # Henüz takip süresi dolmadıysa önceki haliyle tutmaya devam et (görünmese de)
                updated_prev_tracks[track_id] = track_info
                # Not: Görünmeyen nesnenin hızı bu karede HESAPLANMAYACAKTIR.

    # Mevcut karede eşleşmeyen (yani yeni) tespitleri yeni takiplere ata
    for i, current_det in enumerate(current_detections_raw):
        if i not in matched_current_indices:
            new_track_id = next_track_id
            next_track_id += 1

            current_det['matched_track_id'] = new_track_id # Tespite yeni track ID'yi ata

            # Yeni takip bilgisini oluştur
            x1, y1, x2, y2 = current_det['bbox']
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_diag = math.sqrt(bbox_width**2 + bbox_height**2) if bbox_width > 0 and bbox_height > 0 else 0

            updated_prev_tracks[new_track_id] = {
                'bbox': current_det['bbox'],
                'timestamp': current_time_sec,
                'diag': bbox_diag,
                'frames_since_last_seen': 0,
                'class_id': current_det['class_id']
            }

    # Takip listesini bu karedeki güncellenmiş listeyle değiştir
    prev_tracks = updated_prev_tracks

    # --- Hız Hesaplama ve Görselleştirme ---
    annotated_frame = frame.copy()
    current_speeds_kmh = {}

    # Güncel takip edilen her araç için (prev_tracks listesi ve frames_since_last_seen == 0 olanlar) hız hesapla
    # Bu loop sadece bu karede 'görünen' nesneler için çalışır.
    for track_id, track_info in prev_tracks.items():
        if track_info['frames_since_last_seen'] > 0:
             # Bu karede görünmeyen nesneler için hız hesaplamayı atla
             continue

        # Track bilgileri zaten güncel (bu karedeki bbox, zaman, diag)
        xyxy = track_info['bbox']
        current_time = track_info['timestamp']
        current_diag = track_info['diag']
        # class_id = track_info['class_id'] # İsterseniz kullanabilirsiniz

        # Sınırlayıcı kutu koordinatları (Tam sayılara dönüştürülmüş)
        x1, y1, x2, y2 = map(int, xyxy) # map kullanarak int'e dönüştür

        # Referans noktası (alt-orta) - Piksel koordinatları
        cx_pixel = (x1 + x2) // 2
        cy_pixel = y2 # Alt orta nokta
        current_pixel_coord = (cx_pixel, cy_pixel)

        # Mevcut zamanı, konumu ve köşegeni ilgili deque'lere kaydet
        # Deque maxlen=2 olduğu için eski veriler otomatik atılır
        track_history_coords[track_id].append(current_pixel_coord)
        track_history_times[track_id].append(current_time)
        track_history_diags[track_id].append(current_diag)

        speed_kmh = 0 # Varsayılan hız
        distance_adjustment_factor = 1.0 # Uzaklık ayar faktörü

        # Eğer bu araç için yeterli geçmiş veri varsa (en az 2 nokta) hızını hesapla
        if len(track_history_coords[track_id]) >= 2 and len(track_history_times[track_id]) >= 2:
            # Önceki ve mevcut koordinatları/zamanları al
            prev_coord_pix = track_history_coords[track_id][0] # Deque'nin 0. elemanı en eski (önceki)
            curr_coord_pix = track_history_coords[track_id][1] # Deque'nin 1. elemanı en yeni (mevcut)
            prev_time = track_history_times[track_id][0]
            curr_time = track_history_times[track_id][1]

            # Geçen süreyi hesapla (saniye)
            elapsed_time = curr_time - prev_time

            if elapsed_time > 1e-6: # Çok küçük zaman farklarını veya sıfırı engelle
                # Piksel cinsinden mesafeyi hesapla
                pixel_distance = math.dist(prev_coord_pix, curr_coord_pix)

                # Piksel cinsinden hızı hesapla (piksel/s)
                speed_pixels_per_sec = pixel_distance / elapsed_time

                # 2. Gerçek Dünya Hızı Tahmini (Varsayılan Genişlik ile)
                scale_factor_m_per_pix = 0
                # Güncel sınırlayıcı kutu genişliğini kullanıyoruz
                current_bbox_width_pixels = x2 - x1
                if current_bbox_width_pixels > 0:
                   scale_factor_m_per_pix = ASSUMED_CAR_WIDTH_METERS / current_bbox_width_pixels
                else:
                   scale_factor_m_per_pix = 0 # Hesaplama başarısız

                # --- İlk Hız Tahmini (m/s) ---
                speed_mps_initial = 0
                if scale_factor_m_per_pix > 0:
                    speed_mps_initial = speed_pixels_per_sec * scale_factor_m_per_pix

                # 3. Deneysel Uzaklık Ayarı (Köşegen Değişimi ile)
                # Eğer yeterli geçmiş köşegen verisi varsa (en az 2 nokta)
                if len(track_history_diags[track_id]) >= 2:
                    prev_diag = track_history_diags[track_id][0] # Önceki köşegen
                    curr_diag = track_history_diags[track_id][1] # Mevcut köşegen

                    if prev_diag > 0 and curr_diag > 0 and elapsed_time > 1e-6: # Geçerli köşegen değerleri ve geçerli zaman farkı varsa
                        diag_change = curr_diag - prev_diag
                        diag_change_rate = diag_change / elapsed_time # piksel/saniye

                        # Eşiklere göre ayarlama faktörünü belirle
                        if diag_change_rate > DIAG_RATE_THRESHOLD_NEAR: # Hızlı büyüyor -> Yakın
                            distance_adjustment_factor = ADJUST_FACTOR_NEAR
                        elif diag_change_rate < DIAG_RATE_THRESHOLD_FAR: # Küçülüyor/Yavaş büyüyor -> Uzak
                            distance_adjustment_factor = ADJUST_FACTOR_FAR
                        # Diğer durumlar için (orta hızda büyüme) faktör 1.0 kalır

                    else:
                        # Geçerli köşegen yoksa veya 0 ise ayarlama yapma
                        distance_adjustment_factor = 1.0
                else:
                    # Yeterli köşegen geçmişi yoksa ayarlama yapma
                    distance_adjustment_factor = 1.0

                # --- Nihai Hız Hesabı (km/h) ---
                speed_mps_final = speed_mps_initial * distance_adjustment_factor
                speed_kmh = speed_mps_final * 3.6 # m/s to km/h

                # Mantıksız hızları filtrele (isteğe bağlı)
                if speed_kmh < 0: speed_kmh = 0
                if speed_kmh > 180: speed_kmh = 0 # Makul bir üst limit

                current_speeds_kmh[track_id] = speed_kmh # Hızı sakla

        # Görselleştirme için etiket oluştur
        label = "" # Etiketi boş başlatıyoruz, sadece hız varsa doldurulacak
        # Sadece mantıklı hızlar için etikete ekle (örneğin > 5 km/h)
        if track_id in current_speeds_kmh and current_speeds_kmh[track_id] > 5: # Eşiği 5 olarak bıraktım
             label = f"{int(current_speeds_kmh[track_id])} km/h" # Sadece hız bilgisini ata
        # İsteğe bağlı olarak ek bilgiler ekleyebilirsiniz, örneğin sınıf adı:
        # class_name = model.names[track_info['class_id']]
        # if label: # Eğer hız etiketi zaten varsa başına veya sonuna ekle
        #    label = f"{class_name} {label}"
        # else: # Hız etiketi yoksa sadece sınıf adını göster
        #    label = class_name

        # Sınırlayıcı kutuyu çiz
        color = (0, 255, 0) # Yeşil
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Etiket boş değilse çizim yap
        if label:
            # Etiketi çizmek için konum belirle
            label_y_pos = y1 - 10 # Kutunun 10 piksel üstü
            if label_y_pos < 0: # Ekranın dışına çıkarsa kutunun 10 piksel altına al
                 label_y_pos = y2 + 20 # Metin yüksekliği için biraz boşluk

            # Etiketi çiz
            cv2.putText(annotated_frame, label, (x1, label_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # Beyaz metin

    # Artık takip edilmeyen nesnelerin geçmişini temizle (Bellek kullanımı için önemli)
    # prev_tracks sözlüğünde artık olmayan (süresi dolan) track ID'lerinin geçmiş verilerini sil
    ids_to_delete_from_history = [tid for tid in track_history_coords.keys() if tid not in prev_tracks]
    for tid in ids_to_delete_from_history:
        del track_history_coords[tid]
        del track_history_times[tid]
        del track_history_diags[tid]
        # print(f"Track ID {tid} geçmişi temizlendi.") # Debug

    # Gerçek İşleme FPS bilgisini ekle (Basit yaklaşım)
    elapsed_since_start = time.time() - processing_start_time
    actual_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
    cv2.putText(annotated_frame, f"Processing FPS: {actual_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # İsterseniz Video FPS'ini de gösterebilirsiniz:
    # cv2.putText(annotated_frame, f"Video FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Sonucu göster
    cv2.imshow("Hiz Tahmini (YOLOv8 + IoU Takip + Deneysel Ayar)", annotated_frame)

    # Çıkış için tuş kontrolü
    if cv2.waitKey(1) & 0xFF == 27: # ESC tuşuna basılınca çık
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

print("İşlem tamamlandı.")
