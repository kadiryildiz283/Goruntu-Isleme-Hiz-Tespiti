# Gerekli kütüphaneleri içe aktarır:
import cv2  # OpenCV: Görüntü işleme ve video analizi için kullanılır.
import numpy as np  # NumPy: Sayısal işlemler, özellikle dizilerle çalışmak için kullanılır.
from ultralytics import YOLO  # YOLO: Nesne tespiti için kullanılan YOLO modelini içerir.
import math  # Matematiksel işlemler (karekök, trigonometri vb.) için kullanılır.
import time  # Zamanla ilgili işlemler (gecikme, performans ölçümü vb.) için kullanılır.
from collections import defaultdict, deque  # Veri yapıları: defaultdict (varsayılan değerli sözlük) ve deque (çift uçlu kuyruk)

# --- Ayarlar Bölümü ---
# Bu bölümde programın çalışması için gerekli temel parametreler tanımlanır.

VIDEO_PATH = '8.mp4'  # İşlenecek video dosyasının yolu.
YOLO_MODEL = 'yolov8x.pt'  # Kullanılacak YOLOv8 modelinin dosya adı (önceden eğitilmiş bir model). 'x' genellikle en büyük ve en doğru modeldir.
TARGET_CLASSES_IDX = [2, 5, 7]  # YOLO modelinin algılayacağı nesnelerden hangilerinin hedefleneceğini belirten sınıf indeksleri.
                                # COCO veri setinde genellikle: 2=car, 5=bus, 7=truck gibi araç sınıflarını temsil eder.
IOU_THRESHOLD = 0.4  # Intersection over Union (IoU) eşik değeri. Nesne takibinde, bir önceki ve mevcut karedeki kutuların ne kadar
                     # örtüştüğünü belirleyerek aynı nesne olup olmadığına karar vermek için kullanılır.
TRACK_EXPIRY_FRAMES = 20  # Bir nesnenin kaç kare boyunca tespit edilemezse takibinin sonlandırılacağını belirler.
ASSUMED_CAR_WIDTH_METERS = 1.8  # Hız tahmini için varsayılan bir arabanın ortalama genişliği (metre cinsinden).
                                # Bu değer, piksel ölçümlerini gerçek dünya mesafelerine dönüştürmek için kullanılır.
MAX_ANGLE_FOR_WIDTH_CORRECTION_DEG = 75  # Aracın hareket açısına bağlı genişlik düzeltmesinin uygulanacağı maksimum açı (derece).
                                        # Bu açının üzerindeki araçlar için düzeltme faktörü sınırlanır veya uygulanmaz.
MAX_WIDTH_ENLARGEMENT_FACTOR = 2.5  # Açı düzeltmesi sırasında sınırlayıcı kutu genişliğinin ne kadar büyütülebileceğinin üst sınırı.
                                   # Aşırı büyütmeleri engeller.
DIAG_RATE_THRESHOLD_NEAR = 30  # Nesne köşegeninin büyüme hızına (piksel/saniye) göre "yakınlaşma" durumu için eşik değer.
                               # Bu eşiğin üzerindeki pozitif değişim, nesnenin yaklaştığını gösterir ve hız ayarlamasında kullanılır.
DIAG_RATE_THRESHOLD_FAR = -10  # Nesne köşegeninin küçülme hızına (piksel/saniye) göre "uzaklaşma" durumu için eşik değer.
                               # Bu eşiğin altındaki negatif değişim (yani mutlak değerce büyük küçülme), nesnenin uzaklaştığını gösterir.
ADJUST_FACTOR_FAR = 1.5  # Uzaklaşan nesneler için hız tahminine uygulanacak düzeltme faktörü.
ADJUST_FACTOR_NEAR = 0.85  # Yakınlaşan nesneler için hız tahminine uygulanacak düzeltme faktörü.

# --- Yeni Ayarlar ---
LABEL_UPDATE_INTERVAL_SEC = 1.0  # Tespit edilen nesnelerin üzerine yazılan etiketlerin (ID, hız vb.) ekranda güncellenme sıklığı (saniye).
                                # Bu, etiketlerin çok sık değişip titreşmesini engeller.
# --------------------

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

# Takip edilecek nesnelere atanacak bir sonraki benzersiz ID için sayaç.
next_track_id = 0
# Bir önceki karede takip edilen nesnelerin bilgilerini (ID'leri anahtar olacak şekilde) saklayan sözlük.
prev_tracks = {}
# Her bir takip edilen nesnenin (track_id) son 'maxlen' koordinatını saklamak için kullanılır. 'maxlen=2' son iki konumu tutar.
track_history_coords = defaultdict(lambda: deque(maxlen=2))
# Her bir takip edilen nesnenin (track_id) son 'maxlen' zaman damgasını saklamak için kullanılır.
track_history_times = defaultdict(lambda: deque(maxlen=2))
# Her bir takip edilen nesnenin (track_id) son 'maxlen' sınırlayıcı kutu köşegen uzunluğunu saklamak için kullanılır.
track_history_diags = defaultdict(lambda: deque(maxlen=2))
# Her bir takip edilen nesnenin (track_id) son 'maxlen' hareket açısını saklamak için kullanılır. 'maxlen=5' son beş açıyı tutar.
track_history_angles = defaultdict(lambda: deque(maxlen=5))

# --- Etiket Güncelleme için Yeni Veri Yapıları ---
# Her bir track_id için son etiket güncelleme zamanını (saniye cinsinden) saklar. float tipinde varsayılan değer alır.
track_last_label_update_time = defaultdict(float)
# Her bir track_id için önbelleğe alınmış etiket parçalarını (ID, hız gibi metinler) bir liste olarak saklar.
track_cached_label_parts = defaultdict(list)
# -----------------------------------------------

# İşlenen toplam kare sayısını tutar.
frame_count = 0
# Video işlemeye başlama zamanını kaydeder (performans ölçümü için).
processing_start_time = time.time()

# İki sınırlayıcı kutu (bounding box) arasındaki Intersection over Union (IoU) değerini hesaplayan fonksiyon.
def calculate_iou(box1, box2):
    # Kesişim alanının sol üst x koordinatını bulur (iki kutunun x1'lerinin maksimumu).
    x1_i = max(box1[0], box2[0])
    # Kesişim alanının sol üst y koordinatını bulur (iki kutunun y1'lerinin maksimumu).
    y1_i = max(box1[1], box2[1])
    # Kesişim alanının sağ alt x koordinatını bulur (iki kutunun x2'lerinin minimumu).
    x2_i = min(box1[2], box2[2])
    # Kesişim alanının sağ alt y koordinatını bulur (iki kutunun y2'lerinin minimumu).
    y2_i = min(box1[3], box2[3])

    # Kesişim alanının genişliğini hesaplar. Negatifse 0 yapar (kesişim yok).
    inter_width = max(0, x2_i - x1_i)
    # Kesişim alanının yüksekliğini hesaplar. Negatifse 0 yapar (kesişim yok).
    inter_height = max(0, y2_i - y1_i)
    # Kesişim alanını hesaplar.
    inter_area = inter_width * inter_height

    # Birinci kutunun alanını hesaplar.
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    # İkinci kutunun alanını hesaplar.
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Birleşim alanını hesaplar (Alan1 + Alan2 - KesişimAlanı).
    union_area = box1_area + box2_area - inter_area

    # IoU değerini döndürür (KesişimAlanı / BirleşimAlanı). Eğer birleşim alanı 0 ise IoU 0 olur (bölme hatasını önler).
    return inter_area / union_area if union_area > 0 else 0

# Aracın hız vektörünün açısına göre, aracın genişliğinin ne kadarının göründüğünü tahmin etmek için kullanılan
# bir yönelim açısı (phi) hesaplar.
def get_vehicle_orientation_angle_phi(velocity_angle_deg):
    # Eğer hız açısı hesaplanamamışsa (None ise), 0 derece döndürür.
    if velocity_angle_deg is None:
        return 0
    # Hız açısını 0-180 derece aralığına indirger, 90 çıkarır ve mutlak değerini alır.
    # Bu, aracın hareket yönünün yatay eksene ne kadar yakın/uzak olduğunu (0-90 derece arası) bir değere dönüştürür.
    # Örneğin, araç tam yatay (0 veya 180 derece) veya tam dikey (90 derece) hareket ediyorsa, bu değer farklılık gösterir.
    # 0 derece: Araç dikey hareket ediyor (genişliği tam görünüyor olabilir).
    # 90 derece: Araç yatay hareket ediyor (genişliği tam görünüyor olabilir).
    # Amaç, aracın "yanlamasına" ne kadar durduğunu bulmaktır.
    phi_deg = abs((velocity_angle_deg % 180) - 90)
    return phi_deg

# Video karelerini sonsuz bir döngü içinde işlemeye başlar.
while True:
    # Video kaynağından bir sonraki kareyi okur. 'ret' kare okumanın başarılı olup olmadığını, 'frame' ise okunan kareyi içerir.
    ret, frame = cap.read()

    # İşlenen kare sayısını bir artırır.
    frame_count += 1
    # Her 5 kareden sadece birini işler (işlem yükünü azaltmak ve hızı artırmak için).
    # Eğer mevcut kare sayısı 5'in katı değilse, döngünün başına döner ve bir sonraki kareye geçer.
    if frame_count % 5 != 0:
        continue  # Bu kareyi atla, bir sonrakine geç

    # Eğer kare okuma başarısız olduysa (video bitti veya bir hata oluştu).
    if not ret:
        print("Video bitti veya okuma hatası.")
        break  # Döngüden çıkar.

    # Mevcut karenin video içindeki zamanını saniye cinsinden hesaplar.
    current_time_sec = frame_count / fps

    # Yüklenen YOLO modelini kullanarak mevcut karedeki nesneleri tespit eder.
    # verbose=False, modelin konsola detaylı çıktı vermesini engeller. [0] sonucu alırız çünkü tek bir görüntü işliyoruz.
    results = model(frame, verbose=False)[0]
    # Mevcut karede tespit edilen ham (işlenmemiş) nesne bilgilerini saklamak için boş bir liste oluşturur.
    current_detections_raw = []

    # Eğer model sonuçlarında 'boxes' (sınırlayıcı kutular) bilgisi varsa (yani en az bir nesne tespit edilmişse).
    if results.boxes is not None:
        # Tespit edilen her bir nesne için döngüye girer.
        for i in range(len(results.boxes.xyxy)):
            # Sınırlayıcı kutunun koordinatlarını (sol üst x, sol üst y, sağ alt x, sağ alt y) alır ve Python listesine dönüştürür.
            xyxy = results.boxes.xyxy[i].tolist()
            # Tespitin güven skorunu alır.
            confidence = results.boxes.conf[i].item()
            # Tespit edilen nesnenin sınıf ID'sini alır (örneğin, araba için 2).
            class_id = int(results.boxes.cls[i].item())

            # Eğer tespit edilen nesnenin sınıf ID'si, bizim ilgilendiğimiz hedef sınıflardan biriyse.
            if class_id in TARGET_CLASSES_IDX:
                # Nesne bilgilerini (sınırlayıcı kutu, güven skoru, sınıf ID'si ve başlangıçta eşleşen bir track ID'si olmadığı için None)
                # bir sözlük olarak 'current_detections_raw' listesine ekler.
                current_detections_raw.append({
                    'bbox': xyxy, 'confidence': confidence, 'class_id': class_id,
                    'matched_track_id': None  # Başlangıçta hiçbir izle eşleşmedi.
                })

    # Mevcut karedeki algılamalardan hangilerinin bir önceki izle eşleştiğini takip etmek için bir küme (set) oluşturur.
    matched_current_indices = set()
    # Güncellenmiş iz bilgilerini (bir sonraki kare için prev_tracks olacak) tutacak boş bir sözlük.
    updated_prev_tracks = {}

    # Bir önceki karedeki her bir iz (track_id ve track_info) için döngüye girer.
    for track_id, track_info in prev_tracks.items():
        best_match_iou = 0  # Bu iz için en iyi IoU skorunu tutar.
        best_match_idx = -1  # Bu iz için en iyi eşleşen mevcut algılamanın indeksini tutar.

        # Mevcut karedeki her bir ham algılama (current_det) için döngüye girer.
        for i, current_det in enumerate(current_detections_raw):
            # Eğer bu algılama zaten başka bir izle eşleşmişse, onu atlar.
            if i in matched_current_indices:
                continue
            # Önceki izdeki sınırlayıcı kutu ile mevcut algılamadaki sınırlayıcı kutu arasında IoU hesaplar.
            iou = calculate_iou(track_info['bbox'], current_det['bbox'])
            # Eğer hesaplanan IoU, bu iz için şimdiye kadarki en iyi IoU'dan daha iyiyse.
            if iou > best_match_iou:
                best_match_iou = iou  # En iyi IoU'yu günceller.
                best_match_idx = i    # En iyi eşleşen algılamanın indeksini günceller.

        # Eğer en iyi eşleşme IoU'su belirlenen eşik değerinden büyük veya eşitse ve bir eşleşme bulunduysa.
        if best_match_iou >= IOU_THRESHOLD and best_match_idx != -1:
            # Eşleşen mevcut algılamaya bu iz_id'sini atar.
            current_detections_raw[best_match_idx]['matched_track_id'] = track_id
            # Bu algılamanın indeksini eşleşenler kümesine ekler (tekrar kullanılmasın diye).
            matched_current_indices.add(best_match_idx)
            # Eşleşen algılamanın sınırlayıcı kutu koordinatlarını alır.
            x1_curr, y1_curr, x2_curr, y2_curr = current_detections_raw[best_match_idx]['bbox']
            # Sınırlayıcı kutunun genişliğini hesaplar.
            bbox_width = x2_curr - x1_curr
            # Sınırlayıcı kutunun yüksekliğini hesaplar.
            bbox_height = y2_curr - y1_curr
            # Sınırlayıcı kutunun köşegen uzunluğunu hesaplar (Pisagor teoremi). Eğer genişlik veya yükseklik 0 ise köşegen 0 olur.
            bbox_diag = math.sqrt(bbox_width**2 + bbox_height**2) if bbox_width > 0 and bbox_height > 0 else 0

            # Bu izi, güncellenmiş bilgilerle (yeni sınırlayıcı kutu, zaman damgası, köşegen vb.) 'updated_prev_tracks'e ekler.
            updated_prev_tracks[track_id] = {
                'bbox': current_detections_raw[best_match_idx]['bbox'], # Yeni kutu
                'timestamp': current_time_sec,                           # Yeni zaman
                # Yeni köşegen. Eğer yeni köşegen 0 ise ve bir önceki köşegen varsa onu kullanır.
                'diag': bbox_diag if bbox_diag > 0 else track_info.get('diag', 0),
                'frames_since_last_seen': 0,  # Nesne bu karede görüldüğü için sayaç sıfırlanır.
                'class_id': current_detections_raw[best_match_idx]['class_id'] # Sınıf ID'si
            }
        else:
            # Eğer bu iz için mevcut karede yeterli IoU ile bir eşleşme bulunamadıysa.
            # Nesnenin kaç kare boyunca görülmediğini sayan sayacı bir artırır.
            track_info['frames_since_last_seen'] += 1
            # Eğer görülmeme süresi, izin sonlandırılması için belirlenen eşik değerinden küçükse.
            if track_info['frames_since_last_seen'] < TRACK_EXPIRY_FRAMES:
                # Bu izi (görülmemiş olsa da) hala 'updated_prev_tracks' listesinde tutar (belki bir sonraki karede tekrar görülür).
                updated_prev_tracks[track_id] = track_info

    # Mevcut karedeki ham algılamalardan, bir önceki izlerle eşleşmemiş olanlar için döngüye girer (yeni nesneler).
    for i, current_det in enumerate(current_detections_raw):
        if i not in matched_current_indices:  # Eğer bu algılama herhangi bir mevcut izle eşleşmediyse.
            new_track_id = next_track_id  # Yeni bir iz ID'si alır.
            next_track_id += 1            # Bir sonraki kullanılabilir iz ID'sini bir artırır.
            current_det['matched_track_id'] = new_track_id # Algılamaya yeni iz ID'sini atar.

            # Yeni algılamanın sınırlayıcı kutu koordinatlarını alır.
            x1_new, y1_new, x2_new, y2_new = current_det['bbox']
            # Sınırlayıcı kutunun genişliğini hesaplar.
            bbox_width_new = x2_new - x1_new
            # Sınırlayıcı kutunun yüksekliğini hesaplar.
            bbox_height_new = y2_new - y1_new
            # Sınırlayıcı kutunun köşegen uzunluğunu hesaplar.
            bbox_diag_new = math.sqrt(bbox_width_new**2 + bbox_height_new**2) if bbox_width_new > 0 and bbox_height_new > 0 else 0

            # Bu yeni algılama için yeni bir iz oluşturur ve 'updated_prev_tracks'e ekler.
            updated_prev_tracks[new_track_id] = {
                'bbox': current_det['bbox'],
                'timestamp': current_time_sec,
                'diag': bbox_diag_new,
                'frames_since_last_seen': 0, # Yeni iz olduğu için görülmeme sayacı 0.
                'class_id': current_det['class_id']
            }

    # Bir sonraki iterasyon için 'prev_tracks' sözlüğünü, bu karede oluşturulan 'updated_prev_tracks' ile günceller.
    prev_tracks = updated_prev_tracks
    # Üzerine çizim yapmak için mevcut karenin bir kopyasını oluşturur. Orjinal kareyi bozmamak için.
    annotated_frame = frame.copy()

    # Güncellenmiş iz listesindeki (prev_tracks) her bir aktif iz için döngüye girer (hız hesaplama ve çizim için).
    for track_id, track_info in prev_tracks.items():
        # Eğer bu iz bir süredir görülmemişse (yani 'frames_since_last_seen' 0'dan büyükse), bu iz için hız hesaplama vs. yapma.
        if track_info['frames_since_last_seen'] > 0:
            continue # Döngünün bir sonraki adımına geç.

        # İz bilgilerini alır:
        xyxy = track_info['bbox']  # Sınırlayıcı kutu koordinatları.
        current_obj_time = track_info['timestamp']  # Bu nesnenin bu karede algılandığı zaman.
        current_diag = track_info['diag']  # Bu nesnenin bu karedeki köşegen uzunluğu.
        # Sınırlayıcı kutu koordinatlarını tam sayıya dönüştürür (çizim için).
        x1, y1, x2, y2 = map(int, xyxy)
        # Sınırlayıcı kutunun mevcut piksel genişliğini float olarak alır.
        current_bbox_width_pixels = float(x2 - x1)
        # Sınırlayıcı kutunun alt-orta x koordinatını (piksel cinsinden) hesaplar (izleme noktası olarak).
        cx_pixel = (x1 + x2) // 2
        # Sınırlayıcı kutunun alt y koordinatını (piksel cinsinden) hesaplar (izleme noktası olarak).
        cy_pixel = y2 # y2, kutunun altını temsil eder. (y1+y2)//2 de kullanılabilirdi merkez için.
        # Mevcut piksel koordinatını (izleme noktasını) bir demet olarak saklar.
        current_pixel_coord = (cx_pixel, cy_pixel)

        # Bu iz için mevcut koordinatı geçmiş koordinatlar listesine (deque) ekler.
        track_history_coords[track_id].append(current_pixel_coord)
        # Bu iz için mevcut zaman damgasını geçmiş zaman damgaları listesine (deque) ekler.
        track_history_times[track_id].append(current_obj_time)
        # Bu iz için mevcut köşegen uzunluğunu geçmiş köşegenler listesine (deque) ekler.
        track_history_diags[track_id].append(current_diag)

        # Başlangıç değerlerini tanımlar:
        speed_kmh = 0.0  # Tahmini hız (km/saat).
        velocity_angle_deg_smoothed = None  # Yumuşatılmış hız vektörü açısı (derece).
        phi_deg_orientation = 0.0  # Hesaplanan araç yönelim açısı.
        width_correction_factor_applied = 1.0  # Uygulanan genişlik düzeltme faktörü.
        corrected_bbox_width_pixels = current_bbox_width_pixels # Düzeltilmiş sınırlayıcı kutu genişliği (başlangıçta mevcut genişlik).

        # Eğer bu iz için en az iki geçmiş koordinat varsa (yani en az iki karede görüldüyse), hız hesaplanabilir.
        if len(track_history_coords[track_id]) >= 2:
            # Bir önceki ve mevcut koordinatları alır (deque'nin ilk ve ikinci elemanları).
            prev_coord_pix = track_history_coords[track_id][0]
            curr_coord_pix = track_history_coords[track_id][1]
            # Bir önceki ve mevcut zaman damgalarını alır.
            prev_time = track_history_times[track_id][0]
            curr_time = track_history_times[track_id][1]
            # İki zaman damgası arasındaki geçen süreyi (saniye) hesaplar.
            elapsed_time = curr_time - prev_time

            # Eğer geçen süre anlamlı bir değere sahipse (çok küçük değilse, sıfıra bölme hatasını önlemek için).
            if elapsed_time > 0.01: # 10 milisaniyeden büyükse
                # İki koordinat arasındaki Öklid mesafesini (piksel cinsinden) hesaplar.
                pixel_distance = math.dist(prev_coord_pix, curr_coord_pix)
                # X ve Y eksenlerindeki piksel değişimini hesaplar.
                dx = curr_coord_pix[0] - prev_coord_pix[0] # x'teki değişim
                dy = curr_coord_pix[1] - prev_coord_pix[1] # y'deki değişim (görüntü koordinatlarında y aşağı doğru artar)
                # Hareketin radyan cinsinden açısını hesaplar (atan2, doğru çeyreği verir).
                current_velocity_angle_rad = math.atan2(dy, dx)
                # Açıyı dereceye çevirir.
                current_velocity_angle_deg = math.degrees(current_velocity_angle_rad)
                # Açıyı 0-360 derece aralığına normalleştirir.
                current_velocity_angle_deg = (current_velocity_angle_deg + 360) % 360

                # Eğer x veya y yönünde anlamlı bir hareket varsa (titremeleri filtrelemek için).
                if abs(dx) > 1 or abs(dy) > 1:
                    # Mevcut hızı açı geçmişine ekler.
                    track_history_angles[track_id].append(current_velocity_angle_deg)
                # Eğer açı geçmişinde veri varsa.
                if track_history_angles[track_id]:
                    # Yumuşatılmış hız açısı olarak geçmişteki son açıyı alır (daha gelişmiş filtreleme de yapılabilir).
                    velocity_angle_deg_smoothed = track_history_angles[track_id][-1]

                # Eğer yumuşatılmış bir hız açısı varsa ve mevcut sınırlayıcı kutu genişliği 0'dan büyükse.
                if velocity_angle_deg_smoothed is not None and current_bbox_width_pixels > 0:
                    # Araç yönelim açısını (phi) hesaplar.
                    phi_deg_orientation = get_vehicle_orientation_angle_phi(velocity_angle_deg_smoothed)
                    # Eğer yönelim açısı, genişlik düzeltmesi için belirlenen maksimum açıdan küçükse.
                    if phi_deg_orientation < MAX_ANGLE_FOR_WIDTH_CORRECTION_DEG:
                        # Yönelim açısının kosinüsünü hesaplar (radyana çevirerek).
                        cos_phi = math.cos(math.radians(phi_deg_orientation))
                        # Eğer kosinüs değeri çok küçük değilse (sıfıra bölme hatasını önlemek için).
                        if cos_phi > 1e-6: # Yaklaşık sıfır değilse
                            # Genişletme faktörünü hesaplar (1/cos(phi)).
                            enlargement_factor = 1.0 / cos_phi
                            # Genişletme faktörünü sınırlar (minimum 1.0, maksimum MAX_WIDTH_ENLARGEMENT_FACTOR).
                            enlargement_factor = min(max(enlargement_factor, 1.0), MAX_WIDTH_ENLARGEMENT_FACTOR)
                            # Sınırlayıcı kutu genişliğini bu faktöre bölerek düzeltir.
                            # Bu, aracın açılı durmasından kaynaklanan görünür genişlikteki azalmayı telafi etmeye çalışır.
                            # Yani, aracın "gerçek" (kameraya dik duruyormuş gibi) genişliğine yakın bir piksel genişliği elde etmeye çalışır.
                            # Önceki mantık: corrected_bbox_width_pixels = current_bbox_width_pixels / enlargement_factor
                            # Bu, corrected_bbox_width_pixels = current_bbox_width_pixels * cos_phi anlamına gelir.
                            # Yani, açılı duran aracın gözlemlenen genişliği, sanki dik bakılıyormuş gibi olan "etkili" genişliğine küçültülür.
                            # Bu, scale_factor_m_per_pix hesaplamasında kullanılır.
                            corrected_bbox_width_pixels = current_bbox_width_pixels * cos_phi # Düzeltme: orijinal genişliği cos_phi ile çarparak "projeksiyonunu" alırız.
                            # Eğer genişletme faktörü olarak 1/cos_phi kullanılacaksa,
                            # corrected_bbox_width_pixels = current_bbox_width_pixels / enlargement_factor yerine
                            # effective_perpendicular_width_pixels = current_bbox_width_pixels / cos_phi olmalıydı.
                            # Kodda şu anki haliyle `corrected_bbox_width_pixels` daha küçük bir değere dönüşüyor.
                            # `scale_factor_m_per_pix = ASSUMED_CAR_WIDTH_METERS / corrected_bbox_width_pixels` olduğundan,
                            # `corrected_bbox_width_pixels` küçülürse, `scale_factor_m_per_pix` büyür. Bu mantıklı.
                            # Tekrar düzeltilmiş satır:
                            width_correction_factor_applied = enlargement_factor # Uygulanan faktörü saklar.

                # Piksel başına düşen gerçek dünya mesafesini (metre/piksel) hesaplamak için başlangıç değeri.
                scale_factor_m_per_pix = 0
                # Eğer düzeltilmiş sınırlayıcı kutu genişliği 1 pikselden büyükse (anlamlı bir genişlikse).
                if corrected_bbox_width_pixels > 1:
                    # Ölçek faktörünü hesaplar: Varsayılan araç genişliği (metre) / Düzeltilmiş piksel genişliği.
                    scale_factor_m_per_pix = ASSUMED_CAR_WIDTH_METERS / corrected_bbox_width_pixels

                # Başlangıçtaki hızı (metre/saniye) hesaplamak için başlangıç değeri.
                speed_mps_initial = 0
                # Eğer ölçek faktörü hesaplanabildiyse (0'dan büyükse).
                if scale_factor_m_per_pix > 0:
                    # Başlangıç hızını hesaplar: (Piksel mesafesi / Geçen süre) * Ölçek faktörü.
                    speed_mps_initial = (pixel_distance / elapsed_time) * scale_factor_m_per_pix

                # Mesafe ayar faktörü için başlangıç değeri (1.0: değişiklik yok).
                distance_adjustment_factor = 1.0
                # Eğer bu iz için en az iki geçmiş köşegen bilgisi varsa.
                if len(track_history_diags[track_id]) >= 2:
                    # Bir önceki ve mevcut köşegen uzunluklarını alır.
                    prev_diag = track_history_diags[track_id][0]
                    curr_diag = track_history_diags[track_id][1]
                    # Köşegen değişimi için geçen süreyi alır (koordinat ve zaman geçmişiyle tutarlı olmalı).
                    time_for_diag_change = track_history_times[track_id][1] - track_history_times[track_id][0]

                    # Eğer önceki ve mevcut köşegenler pozitifse ve geçen süre anlamlıysa.
                    if prev_diag > 0 and curr_diag > 0 and time_for_diag_change > 0.01:
                        # Köşegen değişim oranını (piksel/saniye) hesaplar.
                        diag_change_rate = (curr_diag - prev_diag) / time_for_diag_change
                        # Eğer köşegen değişim oranı "yakınlaşma" eşiğinden büyükse (nesne hızla büyüyor/yaklaşıyor).
                        if diag_change_rate > DIAG_RATE_THRESHOLD_NEAR:
                            distance_adjustment_factor = ADJUST_FACTOR_NEAR # Hızı azaltan faktörü uygula.
                        # Eğer köşegen değişim oranı "uzaklaşma" eşiğinden küçükse (nesne hızla küçülüyor/uzaklaşıyor).
                        elif diag_change_rate < DIAG_RATE_THRESHOLD_FAR:
                            distance_adjustment_factor = ADJUST_FACTOR_FAR # Hızı artıran faktörü uygula.

                # Nihai hızı (metre/saniye) hesaplar: Başlangıç hızı * Mesafe ayar faktörü.
                speed_mps_final = speed_mps_initial * distance_adjustment_factor
                # Nihai hızı km/saat'e çevirir (1 m/s = 3.6 km/h).
                speed_kmh = speed_mps_final * 3.6
                # Hız negatifse 0 yapar (mantıksız durumu engeller).
                if speed_kmh < 0: speed_kmh = 0
                # Hızı maksimum bir değerle sınırlar (örneğin 200 km/s).
                if speed_kmh > 200: speed_kmh = 199

        # --- Etiket Güncelleme Mantığı ---
        # Amacı: Etiketlerin (ID, hız vb.) çok sık güncellenerek ekranda titreşmesini engellemek.
        # Eğer bu iz için önbellekte etiket yoksa VEYA son güncellemeden bu yana yeterli süre geçtiyse.
        if not track_cached_label_parts[track_id] or \
           (current_time_sec - track_last_label_update_time[track_id] >= LABEL_UPDATE_INTERVAL_SEC):

            # Mevcut etiket parçalarını oluşturmaya başlar (önce ID).
            current_label_parts = [f"ID:{track_id}"]
            # Eğer hesaplanan hız anlamlı bir değerse (0.5 km/s'den büyükse).
            if speed_kmh > 0.5:
                # Hız bilgisini etiket parçalarına ekler.
                current_label_parts.append(f"{int(speed_kmh)}km/h")
            # İsteğe bağlı debug bilgileri (eğer yumuşatılmış hız açısı varsa).
            if velocity_angle_deg_smoothed is not None:
                current_label_parts.append(f"MvAng:{int(velocity_angle_deg_smoothed)}d") # Hareket Açısı
            # Aşağıdaki satırlar yorumlanmış, istenirse açılabilir.
            #current_label_parts.append(f"Phi:{phi_deg_orientation:.0f}d") # Yönelim Açısı
            #if width_correction_factor_applied > 1.01 :
            #    current_label_parts.append(f"WCF:{width_correction_factor_applied:.1f}x") # Genişlik Düzeltme Faktörü

            # Oluşturulan etiket parçalarını bu iz için önbelleğe alır.
            track_cached_label_parts[track_id] = current_label_parts
            # Bu iz için son etiket güncelleme zamanını mevcut zaman olarak kaydeder.
            track_last_label_update_time[track_id] = current_time_sec

        # --- Görselleştirme (Her zaman önbellekten çiz) ---
        # Sınırlayıcı kutunun rengini belirler. Varsayılan yeşil.
        box_color = (0, 255, 0) # Yeşil
        # Eğer yönelim açısı, genişlik düzeltmesinin uygulanmayacağı kadar büyükse, kutu rengini turuncu yapar.
        if phi_deg_orientation >= MAX_ANGLE_FOR_WIDTH_CORRECTION_DEG :
            box_color = (0, 165, 255) # Turuncu
        # Eğer genişlik düzeltme faktörü anlamlı bir şekilde uygulanmışsa (1.05'ten büyükse), kutu rengini kırmızı yapar.
        elif width_correction_factor_applied > 1.05 :
            box_color = (255, 0, 0) # Kırmızı

        # Sınırlayıcı kutuyu (dikdörtgeni) kare üzerine çizer.
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2) # Kalınlık 2

        # Önbellekten alınan etiketleri kare üzerine yazdırmaya başlar.
        # Etiketlerin y eksenindeki başlangıç pozisyonunu belirler (kutunun biraz üstü).
        label_y_start_pos = y1 - 7
        # Eğer bu iz için önbellekte etiket parçaları varsa.
        if track_cached_label_parts[track_id]:
            # Her bir etiket parçası için döngüye girer.
            for i, part_text in enumerate(track_cached_label_parts[track_id]):
                # Metni iki kez çizer: Biri kalın beyaz (arka plan gibi), diğeri renkli (ön plan). Bu, okunabilirliği artırır.
                # Kalın beyaz metin (dış hat için).
                cv2.putText(annotated_frame, part_text, (x1 + 3, label_y_start_pos - (i * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA)
                # Renkli metin (asıl metin).
                cv2.putText(annotated_frame, part_text, (x1 + 3, label_y_start_pos - (i * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # Artık 'prev_tracks' listesinde olmayan (yani takibi sonlandırılmış) izlerin ID'lerini bulur.
    ids_to_delete_from_history = [tid for tid in track_history_coords.keys() if tid not in prev_tracks]
    # Bu eski izlere ait geçmiş verilerini temizler.
    for tid_del in ids_to_delete_from_history:
        if tid_del in track_history_coords: del track_history_coords[tid_del]
        if tid_del in track_history_times: del track_history_times[tid_del]
        if tid_del in track_history_diags: del track_history_diags[tid_del]
        if tid_del in track_history_angles: del track_history_angles[tid_del]
        # Yeni eklenen etiket güncelleme sözlüklerinden de siler.
        if tid_del in track_last_label_update_time: del track_last_label_update_time[tid_del]
        if tid_del in track_cached_label_parts: del track_cached_label_parts[tid_del]

    # Programın başlangıcından bu yana geçen toplam süreyi hesaplar.
    elapsed_since_start = time.time() - processing_start_time
    # Gerçek işleme FPS'ini hesaplar (işlenen kare sayısı / geçen süre).
    actual_processing_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
    # İşleme FPS'ini kare üzerine yazar.
    cv2.putText(annotated_frame, f"Processing FPS: {actual_processing_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # İşlenmiş (üzerine çizimler yapılmış) kareyi bir pencerede gösterir.
    cv2.imshow("Hiz Tahmini (Aci Duzeltmeli v3 - Etiket Seyreltme)", annotated_frame)

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
