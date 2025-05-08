import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from collections import defaultdict, deque

# --- Ayarlar ---
VIDEO_PATH = '8.mp4'  # Video dosyanızın yolu
YOLO_MODEL = 'yolov8x.pt' # Kullanılacak YOLO modeli

# Takip edilecek COCO sınıf indeksleri (car: 2, bus: 5, truck: 7)
# YOLO modelinizin çıktığı sınıflara göre bu indeksleri ayarlayın.
# model.names listesini yazdırarak kontrol edebilirsiniz.
TARGET_CLASSES_IDX = [2, 5, 7]

# --- Basit IoU Takip Ayarları ---
IOU_THRESHOLD = 0.5  # İki sınırlayıcı kutuyu aynı nesneye ait saymak için IoU eşiği
TRACK_EXPIRY_FRAMES = 15 # Bir nesne kaç kare boyunca algılanmazsa takibi bırakılacak

# --- Hız Tahmini Ayarları ---
# Bu değerler VİDEOYA ÖZELDİR ve sahnenize/kameranıza göre AYARLANMALIDIR!
# Hız hesaplaması için kullanılan piksel-metre ölçeği, sınırlayıcı kutu genişliğinin
# varsayılan araç genişliğine (1.8m) karşılık geldiği varsayımına dayanır.
# Bu varsayım, araç kameraya dik olduğunda doğrudur. Açılı araçlarda hız tahmini
# daha az doğru olacaktır. Aracın yönelim açısını 2D kutudan doğrudan çıkarmak zordur.
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
    # Modelin sınıf isimlerini görmek için yorum satırını kaldırın:
    # print("Model Sınıfları:", model.names)
    # print("Takip Edilen Sınıf İndeksleri:", TARGET_CLASSES_IDX)
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

# Takip edilen nesnelerin bilgileri (Track ID -> Bilgiler)
# IoU takibi için kullanılacak ana yapı
prev_tracks = {} # {track_id: {'bbox': [x1, y1, x2, y2], 'timestamp': t, 'diag': diag_pix, 'frames_since_last_seen': n, 'class_id': cls}}

# Araçların önceki PİKSEL konumlarını (alt-orta), zamanlarını ve KÖŞEGENLERİNİ saklamak için
# deque(maxlen=2): Sadece en son 2 değeri tutar (önceki ve mevcut)
# Bu yapı, track_id'lere göre hız, yatay uzunluk ve hareket açısı hesaplama geçmişini tutacak.
track_history_coords = defaultdict(lambda: deque(maxlen=2))
track_history_times = defaultdict(lambda: deque(maxlen=2))
track_history_diags = defaultdict(lambda: deque(maxlen=2)) # Köşegen değişimi için

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
            if class_id in TARGET_CLASSES_IDX:
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

    # Güncel takip edilen her araç için (prev_tracks listesi ve frames_since_last_seen == 0 olanlar) bilgileri işle
    for track_id, track_info in prev_tracks.items():
        if track_info['frames_since_last_seen'] > 0:
             # Bu karede görünmeyen nesneler için işlem yapmayı atla
             continue

        # Track bilgileri zaten güncel (bu karedeki bbox, zaman, diag)
        xyxy = track_info['bbox']
        current_time = track_info['timestamp']
        current_diag = track_info['diag']
        class_id = track_info['class_id']

        # Sınırlayıcı kutu koordinatları (Tam sayılara dönüştürülmüş)
        x1, y1, x2, y2 = map(int, xyxy)

        # Yatay alt çizgi uzunluğu (bounding box genişliği)
        current_bbox_width_pixels = x2 - x1

        # Referans nokta (alt-orta) - Piksel koordinatları
        cx_pixel = (x1 + x2) // 2
        cy_pixel = y2 # Alt orta nokta
        current_pixel_coord = (cx_pixel, cy_pixel)

        # Mevcut zamanı, konumu ve köşegeni ilgili deque'lere kaydet
        # Deque maxlen=2 olduğu için eski veriler otomatik atılır
        track_history_coords[track_id].append(current_pixel_coord)
        track_history_times[track_id].append(current_time)
        track_history_diags[track_id].append(current_diag)

        speed_kmh = 0 # Varsayılan hız
        velocity_angle_deg = None # Varsayılan hareket açısı
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

            if elapsed_time > 0.01: # Çok küçük zaman farklarını veya sıfırı engelle (FPS'ye göre eşiği ayarlayın)
                # Piksel cinsinden mesafeyi hesapla (alt-orta noktalar arası)
                pixel_distance = math.dist(prev_coord_pix, curr_coord_pix)

                # Görüntü düzlemi hareket vektörü ve açısı
                dx = curr_coord_pix[0] - prev_coord_pix[0]
                dy = curr_coord_pix[1] - prev_coord_pix[1]

                # math.atan2(y, x) açıyı radyan cinsinden döndürür
                # y ekseni görüntüde aşağı doğru olduğu için açıyı buna göre yorumlayın veya dy'yi -dy alın.
                # Burada (dx, dy) olarak kullanmak, sağ yatayı 0 derece kabul edip saat yönünde artan bir açı verir.
                velocity_angle_rad = math.atan2(dy, dx)
                velocity_angle_deg = math.degrees(velocity_angle_rad)
                # Açıyı 0-360 derece aralığına normalleştirme
                velocity_angle_deg = (velocity_angle_deg + 360) % 360


                # 2. Gerçek Dünya Hızı Tahmini (Varsayılan Genişlik ve Mevcut Bbox Genişliği ile)
                # Bu ölçek, mevcut bbox genişliğinin 1.8 metreye karşılık geldiğini varsayar.
                # Aracın açısı bu hesaplamanın doğruluğunu etkiler.
                scale_factor_m_per_pix = 0
                if current_bbox_width_pixels > 0:
                   scale_factor_m_per_pix = ASSUMED_CAR_WIDTH_METERS / current_bbox_width_pixels
                else:
                   scale_factor_m_per_pix = 0 # Hesaplama başarısız veya bbox genişliği sıfır

                # --- İlk Hız Tahmini (m/s) ---
                speed_mps_initial = 0
                if scale_factor_m_per_pix > 0:
                    speed_mps_initial = (pixel_distance / elapsed_time) * scale_factor_m_per_pix # (piksel/s) * (m/piksel) = m/s

                # 3. Deneysel Uzaklık Ayarı (Köşegen Değişimi ile)
                # Eğer yeterli geçmiş köşegen verisi varsa (en az 2 nokta)
                if len(track_history_diags[track_id]) >= 2:
                    prev_diag = track_history_diags[track_id][0] # Önceki köşegen
                    curr_diag = track_history_diags[track_id][1] # Mevcut köşegen

                    if prev_diag > 0 and curr_diag > 0 and elapsed_time > 0.01: # Geçerli köşegen değerleri ve geçerli zaman farkı varsa
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
                if speed_kmh > 200: speed_kmh = 0 # Makul bir üst limit (200 km/h)

            else: # Yeterli zaman farkı yoksa hız hesaplanamaz
                 speed_kmh = 0
                 velocity_angle_deg = None


        # Görselleştirme için etiket oluştur
        label = f"ID: {track_id}"
        if speed_kmh > 3: # Sadece hareket eden araçlar için hızı göster (eşik 3 km/h)
            label += f" Speed: {int(speed_kmh)} km/h"
        label += f" Width: {current_bbox_width_pixels} px"
        if velocity_angle_deg is not None:
             label += f" Angle: {int(velocity_angle_deg)} deg" # Hareket açısını etikete ekle


        # Sınırlayıcı kutuyu çiz
        color = (0, 255, 0) # Yeşil
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Etiketi çizmek için konum belirle
        label_y_pos = y1 - 10 # Kutunun 10 piksel üstü
        if label_y_pos < 0: # Ekranın dışına çıkarsa kutunun 10 piksel altına al
             label_y_pos = y2 + 20 # Metin yüksekliği için biraz boşluk

        # Etiketi çiz
        # Birden fazla bilgi satırı için etiketi bölelim
        label_lines = label.split(' ')
        y_offset = 0
        for line in label_lines:
            cv2.putText(annotated_frame, line, (x1, label_y_pos + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # Beyaz metin
            y_offset += 15 # Bir sonraki satır için y konumunu ayarla


    # Artık takip edilmeyen nesnelerin geçmişini temizle (Bellek kullanımı için önemli)
    # prev_tracks sözlüğünde artık olmayan (süresi dolan) track ID'lerinin geçmiş verilerini sil
    ids_to_delete_from_history = [tid for tid in track_history_coords.keys() if tid not in prev_tracks]
    for tid in ids_to_delete_from_history:
        if tid in track_history_coords: del track_history_coords[tid]
        if tid in track_history_times: del track_history_times[tid]
        if tid in track_history_diags: del track_history_diags[tid]
        # print(f"Track ID {tid} geçmişi temizlendi.") # Debug

    # Gerçek İşleme FPS bilgisini ekle
    elapsed_since_start = time.time() - processing_start_time
    actual_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
    cv2.putText(annotated_frame, f"Processing FPS: {actual_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Sonucu göster
    cv2.imshow("Hiz Tahmini (YOLOv8 + IoU Takip + Deneysel Ayar)", annotated_frame)

    # Çıkış için tuş kontrolü
    if cv2.waitKey(1) & 0xFF == 27: # ESC tuşuna basılınca çık
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

print("İşlem tamamlandı.")
