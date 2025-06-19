# Gerekli kütüphaneleri içe aktarır:
import cv2  # OpenCV: Görüntü işleme ve video analizi için kullanılır.
import numpy as np  # NumPy: Sayısal işlemler, özellikle dizilerle çalışmak için kullanılır.
from ultralytics import YOLO  # YOLO: Nesne tespiti için kullanılan YOLO modelini içerir.
import math  # Matematiksel işlemler (tanjant, radyan vb.) için kullanılır.
import time  # Zamanla ilgili işlemler (gecikme, performans ölçümü vb.) için kullanılır.
from collections import defaultdict, deque  # Veri yapıları: defaultdict (varsayılan değerli sözlük) ve deque (çift uçlu kuyruk)

# --- Ayarlar Bölümü ---
# Bu bölümde programın çalışması için gerekli temel parametreler tanımlanır.

VIDEO_PATH = 'top.mp4'  # İşlenecek video dosyasının yolu. (Top içeren bir video ile değiştirilmeli)
YOLO_MODEL = 'yolov8x.pt'  # Kullanılacak YOLOv8 modelinin dosya adı.
# NOT: Bu model araçları tespit eder. Bir topu tespit etmek için
# ya 'sports ball' gibi ilgili bir sınıfı hedeflemeli (eğer modelde varsa)
# ya da top tespiti için eğitilmiş farklı bir model kullanmalısınız.
# Aşağıdaki TARGET_CLASSES_IDX değeri top sınıfının indeksine göre AYARLANMALIDIR!
TARGET_CLASSES_IDX = [32]  # COCO veri setinde 32 genellikle 'sports ball' (spor topu) sınıfıdır.
                           # Kullandığınız modelin top sınıfı indeksi farklı olabilir! Kontrol edin.
IOU_THRESHOLD = 0.4  # Intersection over Union (IoU) eşik değeri. Nesne takibinde kullanılır.
TRACK_EXPIRY_FRAMES = 20  # Bir nesnenin kaç kare boyunca tespit edilemezse takibinin sonlandırılacağını belirler.
LABEL_UPDATE_INTERVAL_SEC = 0.5 # Tespit edilen nesnelerin üzerine yazılan etiketlerin güncellenme sıklığı (saniye).

# --- HFOV ve Mesafe Hesaplaması İçin Yeni Ayarlar ---
BALL_DIAMETER_CM = 10.0 # Topun gerçek çapı (örn: basketbol topu ~22cm, futbol topu ~22cm) - Bu değeri kullanacağınız topun çapına göre ayarlayın.
HFOV_DEG = 76.66       # Kameranın yatay görüş açısı (Derece cinsinden) - Verdiğiniz hesaplamadan alındı.
# -------------------------------------------------

# Araçlara özel ayarlar kaldırıldı:
# ASSUMED_CAR_WIDTH_METERS = 1.8
# MAX_ANGLE_FOR_WIDTH_CORRECTION_DEG = 75
# MAX_WIDTH_ENLARGEMENT_FACTOR = 2.5
# DIAG_RATE_THRESHOLD_NEAR = 30 # Köşegen değişimi bazlı ayarlama kaldırıldı.
# DIAG_RATE_THRESHOLD_FAR = -10 # Köşegen değişimi bazlı ayarlama kaldırıldı.
# ADJUST_FACTOR_FAR = 1.5      # Köşegen değişimi bazlı ayarlama kaldırıldı.
# ADJUST_FACTOR_NEAR = 0.85    # Köşegen değişimi bazlı ayarlama kaldırıldı.


try:
    # Belirtilen YOLO modelini yüklemeye çalışır.
    model = YOLO(YOLO_MODEL)
except Exception as e:
    # Model yüklenirken bir hata oluşursa, hata mesajını yazdırır ve programı sonlandırır.
    print(f"Hata: YOLO modeli yüklenemedi ({YOLO_MODEL}). Hata: {e}")
    exit()

# Belirtilen yoldan video dosyasını açar.
cap = cv2.VideoCapture(VIDEO_PATH)
# Video dosyasının başarıyla açılıp açılmadığını kontrol eder.
if not cap.isOpened():
    # Video açılamazsa hata mesajı yazdırır ve programı sonlandırır.
    print(f"Hata: Video dosyası açılamadı ({VIDEO_PATH}).")
    exit()

# Videonun saniyedeki kare sayısını (FPS) alır.
fps = cap.get(cv2.CAP_PROP_FPS)
# Eğer FPS değeri alınamazsa (bazı video formatlarında veya kameralarda olabilir), varsayılan olarak 30 FPS kullanılır.
if fps == 0:
    fps = 30
    print(f"Uyarı: FPS alınamadı, varsayılan {fps} kullanılıyor.")
# Videonun genişliğini alır ve tam sayıya dönüştürür.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# Videonun yüksekliğini alır ve tam sayıya dönüştürür.
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Video bilgilerini (dosya yolu, FPS, boyut) ekrana yazdırır.
print(f"Video: {VIDEO_PATH}, FPS: {fps:.2f}, Boyut: {frame_width}x{frame_height}")

# Piksel başına düşen açısal çözünürlüğü radyan cinsinden hesapla (HFOV yöntemi için gerekli)
# HFOV_DEG değeri frame_width pikseli kapsar.
# angle_per_pixel_rad = (HFOV_DEG * math.pi / 180) / frame_width
# Basitlik için önce derece/piksel sonra radyan/piksel hesaplayalım:
angle_per_pixel_deg = HFOV_DEG / frame_width
angle_per_pixel_rad = math.radians(angle_per_pixel_deg)


# Takip edilecek nesnelere atanacak bir sonraki benzersiz ID için sayaç.
next_track_id = 0
# Bir önceki karede takip edilen nesnelerin bilgilerini (ID'leri anahtar olacak şekilde) saklayan sözlük.
prev_tracks = {}

# Her bir takip edilen nesnenin geçmiş verilerini saklamak için deque'ler:
# Son 'maxlen' koordinatı (piksel)
track_history_coords = defaultdict(lambda: deque(maxlen=2))
# Son 'maxlen' zaman damgası (saniye)
track_history_times = defaultdict(lambda: deque(maxlen=2))
# Son 'maxlen' sınırlayıcı kutu köşegen uzunluğu (piksel) - Kaldırıldı ama örnekte duruyordu, tutulabilir veya tamamen çıkarılabilir
# track_history_diags = defaultdict(lambda: deque(maxlen=2))
# Son 'maxlen' HFOV yöntemiyle hesaplanan mesafe (cm)
track_history_distances = defaultdict(lambda: deque(maxlen=2))

# Araç açısı geçmişi kaldırıldı:
# track_history_angles = defaultdict(lambda: deque(maxlen=5))

# Etiket Güncelleme için Veri Yapıları:
track_last_label_update_time = defaultdict(float)
track_cached_label_parts = defaultdict(list)
track_current_speed_kmh = defaultdict(float) # Her iz için son hesaplanan hızı saklar
track_current_distance_cm = defaultdict(float) # Her iz için son hesaplanan mesafeyi saklar

# İki sınırlayıcı kutu (bounding box) arasındaki Intersection over Union (IoU) değerini hesaplayan fonksiyon.
def calculate_iou(box1, box2):
    # Kesişim alanının koordinatlarını bulur
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    # Kesişim alanının genişlik ve yüksekliğini hesaplar. Negatifse 0 yapar.
    inter_width = max(0, x2_i - x1_i)
    inter_height = max(0, y2_i - y1_i)
    inter_area = inter_width * inter_height

    # Kutuların alanlarını hesaplar.
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Birleşim alanını hesaplar.
    union_area = box1_area + box2_area - inter_area

    # IoU değerini döndürür.
    return inter_area / union_area if union_area > 0 else 0

# HFOV yöntemini kullanarak nesnenin mesafesini hesaplayan fonksiyon.
def calculate_distance_cm_hfov(object_pixel_width, real_object_diameter_cm, hfov_deg, total_horizontal_pixels):
    # Nesnenin piksel genişliği 0 veya negatifse mesafe hesaplanamaz.
    if object_pixel_width <= 0:
        return 0.0 # Veya sonsuzluk veya hata değeri dönebilir.

    # Piksel başına düşen açısal çözünürlüğü radyan cinsinden hesapla
    # angle_per_pixel_rad = math.radians(hfov_deg) / total_horizontal_pixels # Bu önceden hesaplandı
    global angle_per_pixel_rad # Global değişkeni kullan

    # Nesnenin görüntüde kapladığı toplam açısal genişliği radyan cinsinden hesapla
    theta_object_rad = object_pixel_width * angle_per_pixel_rad

    # Mesafe hesaplama formülü: D = W_gercek / (2 * tan(theta_object / 2))
    # tan(theta_object / 2) çok küçükse (uzak nesne) veya sıfırsa (piksel genişliği sıfır) bölme hatasını önle
    tan_half_theta = math.tan(theta_object_rad / 2.0)

    if tan_half_theta > 1e-6: # Çok küçük bir değere bölmeyi önle
        distance_cm = real_object_diameter_cm / (2.0 * tan_half_theta)
        return distance_cm
    else:
        return 0.0 # Çok uzak veya piksel genişliği çok küçükse mesafe 0 kabul edilebilir veya NaN dönebilir.


# Araç yönelim açısı fonksiyonu kaldırıldı:
# def get_vehicle_orientation_angle_phi(velocity_angle_deg): ...

# Video karelerini sonsuz bir döngü içinde işlemeye başlar.
while True:
    # Video kaynağından bir sonraki kareyi okur.
    ret, frame = cap.read()

    # İşlenen kare sayısını bir artırır.
    frame_count += 1
    # Her N kareden sadece birini işler (işlem yükünü azaltmak ve hızı artırmak için).
    # Eğer mevcut kare sayısı N'in katı değilse, döngünün başına döner.
    # NOT: HFOV tabanlı mesafe ve hız hesaplaması, kare atlandığında daha uzun zaman aralıklarını kullanır, bu da hızı etkileyebilir.
    # Daha hassas hız için burada atlama oranını azaltabilir veya kaldırabilirsiniz.
    if frame_count % 5 != 0:
        # Ancak, etiket güncelleme mantığı çerçeve atlama oranından bağımsız çalışacaktır.
        # Sadece önceki karedeki takip bilgilerini kullan, yeni tespit yapma
        # Mevcut karenin zamanını güncelle
        current_time_sec = frame_count / fps
        # Sadece görselleştirme ve etiket güncelleme kısmını çalıştır.
        pass # Aşağıdaki görselleştirme kısmı her karede çalışacak şekilde ayarlandı
    else:
        # Eğer kare okuma başarısız olduysa (video bitti veya bir hata oluştu).
        if not ret:
            print("Video bitti veya okuma hatası.")
            break  # Döngüden çıkar.

        # Mevcut karenin video içindeki zamanını saniye cinsinden hesaplar.
        current_time_sec = frame_count / fps

        # Yüklenen YOLO modelini kullanarak mevcut karedeki nesneleri tespit eder.
        results = model(frame, verbose=False)[0]
        # Mevcut karede tespit edilen ham nesne bilgilerini saklamak için boş bir liste oluşturur.
        current_detections_raw = []

        # Eğer model sonuçlarında 'boxes' bilgisi varsa.
        if results.boxes is not None:
            # Tespit edilen her bir nesne için döngüye girer.
            for i in range(len(results.boxes.xyxy)):
                # Sınırlayıcı kutu koordinatları, güven skoru ve sınıf ID'si alınır.
                xyxy = results.boxes.xyxy[i].tolist()
                confidence = results.boxes.conf[i].item()
                class_id = int(results.boxes.cls[i].item())

                # Eğer tespit edilen nesnenin sınıf ID'si, bizim ilgilendiğimiz hedef sınıflardan biriyse.
                if class_id in TARGET_CLASSES_IDX:
                    # Nesne bilgilerini listeye ekler.
                    current_detections_raw.append({
                        'bbox': xyxy, 'confidence': confidence, 'class_id': class_id,
                        'matched_track_id': None  # Başlangıçta hiçbir izle eşleşmedi.
                    })

        # Mevcut karedeki algılamalardan hangilerinin bir önceki izle eşleştiğini takip etmek için bir küme oluşturur.
        matched_current_indices = set()
        # Güncellenmiş iz bilgilerini tutacak boş bir sözlük.
        updated_prev_tracks = {}

        # Bir önceki karedeki her bir iz için döngüye girer (takip ve güncelleme).
        for track_id, track_info in prev_tracks.items():
            best_match_iou = 0  # En iyi IoU skorunu tutar.
            best_match_idx = -1 # En iyi eşleşen mevcut algılamanın indeksini tutar.

            # Mevcut karedeki her bir ham algılama için döngüye girer.
            for i, current_det in enumerate(current_detections_raw):
                # Eğer bu algılama zaten başka bir izle eşleşmişse, atla.
                if i in matched_current_indices:
                    continue
                # IoU hesapla.
                iou = calculate_iou(track_info['bbox'], current_det['bbox'])
                # En iyi eşleşmeyi bul.
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = i

            # Eğer en iyi eşleşme IoU'su belirlenen eşik değerinden büyük veya eşitse ve bir eşleşme bulunduysa.
            if best_match_iou >= IOU_THRESHOLD and best_match_idx != -1:
                # Eşleşen algılamaya bu iz ID'sini atar.
                current_detections_raw[best_match_idx]['matched_track_id'] = track_id
                # Bu algılamanın indeksini eşleşenler kümesine ekler.
                matched_current_indices.add(best_match_idx)

                # Eşleşen algılamanın sınırlayıcı kutu koordinatlarını alır.
                x1_curr, y1_curr, x2_curr, y2_curr = current_detections_raw[best_match_idx]['bbox']
                # Sınırlayıcı kutunun piksel genişliğini hesaplar.
                current_bbox_width_pixels = float(x2_curr - x1_curr)
                # Sınırlayıcı kutunun alt-orta x ve alt y koordinatlarını alır (izleme noktası için).
                cx_pixel = (x1_curr + x2_curr) // 2
                cy_pixel = y2_curr # veya (y1_curr + y2_curr) // 2 orta nokta için

                # HFOV Yöntemi ile Mesafeyi Hesapla
                current_distance_cm = calculate_distance_cm_hfov(
                    current_bbox_width_pixels, BALL_DIAMETER_CM, HFOV_DEG, frame_width
                )

                # İz bilgilerini güncelle
                updated_prev_tracks[track_id] = {
                    'bbox': current_detections_raw[best_match_idx]['bbox'],
                    'timestamp': current_time_sec,
                    # 'diag' bilgisi HFOV yöntemiyle doğrudan kullanılmıyor, isterseniz kaldırabilirsiniz.
                    'diag': math.sqrt(current_bbox_width_pixels**2 + (y2_curr - y1_curr)**2) if current_bbox_width_pixels > 0 and (y2_curr-y1_curr) > 0 else 0,
                    'frames_since_last_seen': 0,
                    'class_id': current_detections_raw[best_match_idx]['class_id'],
                    'distance_cm': current_distance_cm # Yeni: Hesaplanan mesafeyi sakla
                }

                # Geçmiş verilerini güncelle
                track_history_coords[track_id].append((cx_pixel, cy_pixel))
                track_history_times[track_id].append(current_obj_time)
                # track_history_diags[track_id].append(updated_prev_tracks[track_id]['diag']) # İsterseniz köşegen geçmişini de güncelleyin
                track_history_distances[track_id].append(current_distance_cm) # Yeni: Mesafe geçmişini güncelle

                # --- Hız Hesaplama (Mesafe Değişiminden) ---
                speed_kmh = 0.0 # Varsayılan hız
                # Eğer mesafe geçmişinde en az iki kayıt varsa
                if len(track_history_distances[track_id]) >= 2:
                    prev_distance_cm = track_history_distances[track_id][0]
                    curr_distance_cm = track_history_distances[track_id][1]
                    prev_time = track_history_times[track_id][0]
                    curr_time = track_history_times[track_id][1]

                    elapsed_time = curr_time - prev_time

                    # Eğer geçen süre anlamlıysa ve mesafeler hesaplanabildiyse
                    if elapsed_time > 0.01 and prev_distance_cm > 0.0 and curr_distance_cm > 0.0:
                        # Mesafedeki değişimi (cm) hesapla
                        delta_distance_cm = curr_distance_cm - prev_distance_cm
                        # Hızı cm/s cinsinden hesapla (mesafedeki değişim / geçen süre)
                        speed_cmps = delta_distance_cm / elapsed_time

                        # Hızı km/saat'e çevir (1 cm/s = 0.036 km/h)
                        # Hızın mutlak değerini alıyoruz, çünkü yaklaşma/uzaklaşma burada delta_distance'ın işaretinde.
                        speed_kmh = abs(speed_cmps) * 0.036

                # Hesaplanan hızı sakla
                track_current_speed_kmh[track_id] = speed_kmh
                # Hesaplanan mesafeyi sakla
                track_current_distance_cm[track_id] = current_distance_cm


            else:
                # Eğer bu iz için mevcut karede yeterli IoU ile bir eşleşme bulunamadıysa.
                # Nesnenin kaç kare boyunca görülmediğini sayan sayacı bir artırır.
                track_info['frames_since_last_seen'] += 1
                # Eğer görülmeme süresi, izin sonlandırılması için belirlenen eşik değerinden küçükse.
                if track_info['frames_since_last_seen'] < TRACK_EXPIRY_FRAMES:
                    # Bu izi (görülmemiş olsa da) hala 'updated_prev_tracks' listesinde tutar.
                    updated_prev_tracks[track_id] = track_info
                else:
                    # İz süresi dolmuşsa, bu izle ilgili saklanmış hız ve mesafe bilgilerini temizle.
                    if track_id in track_current_speed_kmh: del track_current_speed_kmh[track_id]
                    if track_id in track_current_distance_cm: del track_current_distance_cm[track_id]


        # Mevcut karedeki ham algılamalardan, bir önceki izlerle eşleşmemiş olanlar için döngüye girer (yeni nesneler).
        for i, current_det in enumerate(current_detections_raw):
            if i not in matched_current_indices:  # Eğer bu algılama herhangi bir mevcut izle eşleşmediyse.
                new_track_id = next_track_id  # Yeni bir iz ID'si alır.
                next_track_id += 1            # Bir sonraki kullanılabilir iz ID'sini bir artırır.
                current_det['matched_track_id'] = new_track_id # Algılamaya yeni iz ID'sini atar.

                # Yeni algılamanın sınırlayıcı kutu koordinatlarını alır.
                x1_new, y1_new, x2_new, y2_new = current_det['bbox']
                # Sınırlayıcı kutunun piksel genişliğini hesaplar.
                current_bbox_width_pixels_new = float(x2_new - x1_new)
                # Sınırlayıcı kutunun alt-orta x ve alt y koordinatlarını alır.
                cx_pixel_new = (x1_new + x2_new) // 2
                cy_pixel_new = y2_new # veya (y1_new + y2_new) // 2

                # HFOV Yöntemi ile Başlangıç Mesafesini Hesapla
                initial_distance_cm = calculate_distance_cm_hfov(
                    current_bbox_width_pixels_new, BALL_DIAMETER_CM, HFOV_DEG, frame_width
                )

                # Bu yeni algılama için yeni bir iz oluşturur ve 'updated_prev_tracks'e ekler.
                updated_prev_tracks[new_track_id] = {
                    'bbox': current_det['bbox'],
                    'timestamp': current_time_sec,
                    # 'diag' bilgisi HFOV yöntemiyle doğrudan kullanılmıyor.
                    'diag': math.sqrt(current_bbox_width_pixels_new**2 + (y2_new - y1_new)**2) if current_bbox_width_pixels_new > 0 and (y2_new - y1_new) > 0 else 0,
                    'frames_since_last_seen': 0,
                    'class_id': current_det['class_id'],
                    'distance_cm': initial_distance_cm # Yeni: Başlangıç mesafesini sakla
                }

                # Geçmiş verilerine başlangıç değerlerini ekle
                track_history_coords[new_track_id].append((cx_pixel_new, cy_pixel_new))
                track_history_times[new_track_id].append(current_time_sec)
                # track_history_diags[new_track_id].append(updated_prev_tracks[new_track_id]['diag']) # İsterseniz köşegen geçmişini de ekleyin
                track_history_distances[new_track_id].append(initial_distance_cm) # Yeni: Mesafe geçmişini ekle

        # Bir sonraki iterasyon için 'prev_tracks' sözlüğünü günceller.
        prev_tracks = updated_prev_tracks

    # --- Görselleştirme (Her Karede Çalışır) ---
    # Eğer kare işlendi (frame_count % 5 == 0) ise yeni algılamalar ve güncellemeler yapıldı.
    # Eğer kare atlandıysa, sadece prev_tracks'deki son bilinen konumları ve hızı/mesafeyi kullanır.
    # annotated_frame = frame.copy() # Her karede yenisi oluşturulur (ya da atlanan karede önceki frame kullanılır)
    if frame_count % 5 != 0 and 'annotated_frame' in locals():
        # Kare atlandıysa, sadece önceki çizimleri tutan kareyi al
        pass # annotated_frame zaten önceki döngü adımından kalma
    else:
        # Yeni kare işlendiyse veya ilk kareyse, frame'in kopyasını al
        annotated_frame = frame.copy()


    # Güncellenmiş iz listesindeki (prev_tracks) her bir aktif iz için döngüye girer (çizim için).
    # Not: Hız ve mesafe hesaplaması sadece her 5 karede bir yapılır, ancak etiket gösterimi ve kutular her karede çizilir.
    for track_id, track_info in prev_tracks.items():
        # Eğer bu iz bir süredir görülmemişse, çizim yapma.
        if track_info['frames_since_last_seen'] > 0:
            continue

        # İz bilgilerini alır:
        xyxy = track_info['bbox']  # Sınırlayıcı kutu koordinatları.
        # Sınırlayıcı kutu koordinatlarını tam sayıya dönüştürür (çizim için).
        x1, y1, x2, y2 = map(int, xyxy)

        # --- Etiket Güncelleme ve Çizim Mantığı ---
        # Amacı: Etiketlerin çok sık güncellenerek ekranda titreşmesini engellemek.
        # Eğer bu iz için önbellekte etiket yoksa VEYA son güncellemeden bu yana yeterli süre geçtiyse.
        if not track_cached_label_parts[track_id] or \
           (current_time_sec - track_last_label_update_time[track_id] >= LABEL_UPDATE_INTERVAL_SEC):

            # Mevcut etiket parçalarını oluşturur.
            current_label_parts = [f"ID:{track_id}"]

            # Hesaplanan hızı (km/s) etiket parçalarına ekler (varsa ve anlamlıysa).
            # track_current_speed_kmh sözlüğünden en son hesaplanan hızı alırız.
            current_speed = track_current_speed_kmh.get(track_id, 0.0)
            if abs(current_speed) > 0.5: # Hız 0.5 km/h'ten büyükse göster
                current_label_parts.append(f"Hiz:{int(abs(current_speed))} km/h") # Mutlak hız gösterilir

            # Hesaplanan mesafeyi (cm) etiket parçalarına ekler (varsa ve anlamlıysa).
            # track_current_distance_cm sözlüğünden en son hesaplanan mesafeyi alırız.
            current_distance = track_current_distance_cm.get(track_id, 0.0)
            if current_distance > 0.0: # Mesafe 0'dan büyükse göster
                current_label_parts.append(f"Mesafe:{int(current_distance)} cm")


            # Oluşturulan etiket parçalarını bu iz için önbelleğe alır.
            track_cached_label_parts[track_id] = current_label_parts
            # Bu iz için son etiket güncelleme zamanını mevcut zaman olarak kaydeder.
            track_last_label_update_time[track_id] = current_time_sec

        # Sınırlayıcı kutunun rengini belirler. Örneğin, mesafe 50 cm'den az ise kırmızı, diğer durumlarda yeşil.
        current_distance = track_current_distance_cm.get(track_id, 0.0)
        if current_distance > 0 and current_distance < 50:
            box_color = (0, 0, 255) # Kırmızı
        else:
            box_color = (0, 255, 0) # Yeşil

        # Sınırlayıcı kutuyu kare üzerine çizer.
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2) # Kalınlık 2

        # Önbellekten alınan etiketleri kare üzerine yazdırmaya başlar.
        label_y_start_pos = y1 - 7
        if track_cached_label_parts[track_id]:
            for i, part_text in enumerate(track_cached_label_parts[track_id]):
                # Metni iki kez çizer: Biri kalın beyaz (arka plan gibi), diğeri renkli (ön plan).
                cv2.putText(annotated_frame, part_text, (x1 + 3, label_y_start_pos - (i * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA) # Beyaz dış hat
                cv2.putText(annotated_frame, part_text, (x1 + 3, label_y_start_pos - (i * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA) # Renkli metin


    # Artık 'prev_tracks' listesinde olmayan (yani takibi sonlandırılmış) izlerin ID'lerini bulur.
    ids_to_delete_from_history = [tid for tid in track_history_coords.keys() if tid not in prev_tracks]
    # Bu eski izlere ait geçmiş verilerini temizler.
    for tid_del in ids_to_delete_from_history:
        if tid_del in track_history_coords: del track_history_coords[tid_del]
        if tid_del in track_history_times: del track_history_times[tid_del]
        # if tid_del in track_history_diags: del track_history_diags[tid_del] # Köşegen geçmişi temizliği (kullanmıyorsanız kaldırın)
        if tid_del in track_history_distances: del track_history_distances[tid_del] # Yeni: Mesafe geçmişi temizliği
        # Araç açısı geçmişi temizliği kaldırıldı.
        if tid_del in track_last_label_update_time: del track_last_label_update_time[tid_del]
        if tid_del in track_cached_label_parts: del track_cached_label_parts[tid_del]
        # Güncel hız ve mesafe saklanan sözlüklerden temizle.
        if tid_del in track_current_speed_kmh: del track_current_speed_kmh[tid_del]
        if tid_del in track_current_distance_cm: del track_current_distance_cm[tid_del]


    # Programın başlangıcından bu yana geçen toplam süreyi hesaplar.
    elapsed_since_start = time.time() - processing_start_time
    # Gerçek işleme FPS'ini hesaplar (işlenen kare sayısı / geçen süre).
    # Not: Kare atlama kullanılıyorsa bu değer sadece işlenen kare sayısını yansıtır.
    actual_processing_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
    # İşleme FPS'ini kare üzerine yazar.
    cv2.putText(annotated_frame, f"Processing FPS: {actual_processing_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # İşlenmiş (üzerine çizimler yapılmış) kareyi bir pencerede gösterir.
    cv2.imshow("Top Mesafe ve Hiz Tahmini (HFOV)", annotated_frame)

    # Kullanıcının bir tuşa basmasını 1 milisaniye boyunca bekler.
    # Eğer basılan tuş ESC ise (ASCII değeri 27), döngüden çıkar.
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Video yakalama nesnesini serbest bırakır.
cap.release()
# Açık olan tüm OpenCV pencerelerini kapatır.
cv2.destroyAllWindows()
# İşlemin tamamlandığını belirten bir mesaj yazdırır.
print("İşlem tamamlandı.")
