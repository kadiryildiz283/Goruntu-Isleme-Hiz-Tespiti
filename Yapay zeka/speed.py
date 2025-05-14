import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from collections import defaultdict, deque

# --- Ayarlar ---
VIDEO_PATH = '8.mp4'
YOLO_MODEL = 'yolov8x.pt'
TARGET_CLASSES_IDX = [2, 5, 7]
IOU_THRESHOLD = 0.4
TRACK_EXPIRY_FRAMES = 20
ASSUMED_CAR_WIDTH_METERS = 1.8
MAX_ANGLE_FOR_WIDTH_CORRECTION_DEG = 75
MAX_WIDTH_ENLARGEMENT_FACTOR = 2.5
DIAG_RATE_THRESHOLD_NEAR = 30
DIAG_RATE_THRESHOLD_FAR = -10
ADJUST_FACTOR_FAR = 1.5
ADJUST_FACTOR_NEAR = 0.85

# --- Yeni Ayarlar ---
LABEL_UPDATE_INTERVAL_SEC = 1.0 # Etiketlerin ekranda güncellenme sıklığı (saniye)
# --------------------

try:
    model = YOLO(YOLO_MODEL)
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
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {VIDEO_PATH}, FPS: {fps:.2f}, Boyut: {frame_width}x{frame_height}")

next_track_id = 0
prev_tracks = {}
track_history_coords = defaultdict(lambda: deque(maxlen=2))
track_history_times = defaultdict(lambda: deque(maxlen=2))
track_history_diags = defaultdict(lambda: deque(maxlen=2))
track_history_angles = defaultdict(lambda: deque(maxlen=5))

# --- Etiket Güncelleme için Yeni Veri Yapıları ---
track_last_label_update_time = defaultdict(float) # Her track_id için son etiket güncelleme zamanı
track_cached_label_parts = defaultdict(list)    # Her track_id için önbelleğe alınmış etiket parçaları
# -----------------------------------------------

frame_count = 0
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

def get_vehicle_orientation_angle_phi(velocity_angle_deg):
    if velocity_angle_deg is None:
        return 0
    phi_deg = abs((velocity_angle_deg % 180) - 90)
    return phi_deg

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okuma hatası.")
        break

    frame_count += 1
    current_time_sec = frame_count / fps # Video zamanı bazında saniye

    results = model(frame, verbose=False)[0]
    current_detections_raw = []
    if results.boxes is not None:
        for i in range(len(results.boxes.xyxy)):
            xyxy = results.boxes.xyxy[i].tolist()
            confidence = results.boxes.conf[i].item()
            class_id = int(results.boxes.cls[i].item())
            if class_id in TARGET_CLASSES_IDX:
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
            x1_curr, y1_curr, x2_curr, y2_curr = current_detections_raw[best_match_idx]['bbox']
            bbox_width = x2_curr - x1_curr
            bbox_height = y2_curr - y1_curr
            bbox_diag = math.sqrt(bbox_width**2 + bbox_height**2) if bbox_width > 0 and bbox_height > 0 else 0
            updated_prev_tracks[track_id] = {
                'bbox': current_detections_raw[best_match_idx]['bbox'],
                'timestamp': current_time_sec,
                'diag': bbox_diag if bbox_diag > 0 else track_info.get('diag', 0),
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
            x1_new, y1_new, x2_new, y2_new = current_det['bbox']
            bbox_width_new = x2_new - x1_new
            bbox_height_new = y2_new - y1_new
            bbox_diag_new = math.sqrt(bbox_width_new**2 + bbox_height_new**2) if bbox_width_new > 0 and bbox_height_new > 0 else 0
            updated_prev_tracks[new_track_id] = {
                'bbox': current_det['bbox'],
                'timestamp': current_time_sec,
                'diag': bbox_diag_new,
                'frames_since_last_seen': 0,
                'class_id': current_det['class_id']
            }
    prev_tracks = updated_prev_tracks
    annotated_frame = frame.copy()

    for track_id, track_info in prev_tracks.items():
        if track_info['frames_since_last_seen'] > 0:
            continue

        xyxy = track_info['bbox']
        current_obj_time = track_info['timestamp'] # Bu nesnenin algılandığı zaman
        current_diag = track_info['diag']
        x1, y1, x2, y2 = map(int, xyxy)
        current_bbox_width_pixels = float(x2 - x1)
        cx_pixel = (x1 + x2) // 2
        cy_pixel = y2
        current_pixel_coord = (cx_pixel, cy_pixel)

        track_history_coords[track_id].append(current_pixel_coord)
        track_history_times[track_id].append(current_obj_time) # current_obj_time kullanılmalı
        track_history_diags[track_id].append(current_diag)

        speed_kmh = 0.0
        velocity_angle_deg_smoothed = None
        phi_deg_orientation = 0.0
        width_correction_factor_applied = 1.0
        corrected_bbox_width_pixels = current_bbox_width_pixels

        if len(track_history_coords[track_id]) >= 2:
            prev_coord_pix = track_history_coords[track_id][0]
            curr_coord_pix = track_history_coords[track_id][1]
            prev_time = track_history_times[track_id][0]
            curr_time = track_history_times[track_id][1]
            elapsed_time = curr_time - prev_time

            if elapsed_time > 0.01:
                pixel_distance = math.dist(prev_coord_pix, curr_coord_pix)
                dx = curr_coord_pix[0] - prev_coord_pix[0]
                dy = curr_coord_pix[1] - prev_coord_pix[1]
                current_velocity_angle_rad = math.atan2(dy, dx)
                current_velocity_angle_deg = math.degrees(current_velocity_angle_rad)
                current_velocity_angle_deg = (current_velocity_angle_deg + 360) % 360

                if abs(dx) > 1 or abs(dy) > 1:
                    track_history_angles[track_id].append(current_velocity_angle_deg)
                if track_history_angles[track_id]:
                    velocity_angle_deg_smoothed = track_history_angles[track_id][-1]

                if velocity_angle_deg_smoothed is not None and current_bbox_width_pixels > 0:
                    phi_deg_orientation = get_vehicle_orientation_angle_phi(velocity_angle_deg_smoothed)
                    if phi_deg_orientation < MAX_ANGLE_FOR_WIDTH_CORRECTION_DEG:
                        cos_phi = math.cos(math.radians(phi_deg_orientation))
                        if cos_phi > 1e-6:
                            enlargement_factor = 1.0 / cos_phi
                            enlargement_factor = min(max(enlargement_factor, 1.0), MAX_WIDTH_ENLARGEMENT_FACTOR)
                            corrected_bbox_width_pixels = current_bbox_width_pixels / enlargement_factor
                            width_correction_factor_applied = enlargement_factor
                
                scale_factor_m_per_pix = 0
                if corrected_bbox_width_pixels > 1:
                    scale_factor_m_per_pix = ASSUMED_CAR_WIDTH_METERS / corrected_bbox_width_pixels
                
                speed_mps_initial = 0
                if scale_factor_m_per_pix > 0:
                    speed_mps_initial = (pixel_distance / elapsed_time) * scale_factor_m_per_pix

                distance_adjustment_factor = 1.0
                if len(track_history_diags[track_id]) >= 2:
                    prev_diag = track_history_diags[track_id][0]
                    curr_diag = track_history_diags[track_id][1]
                    # elapsed_time burada da kullanılmalı (track_history_times ile tutarlı)
                    time_for_diag_change = track_history_times[track_id][1] - track_history_times[track_id][0]
                    if prev_diag > 0 and curr_diag > 0 and time_for_diag_change > 0.01:
                        diag_change_rate = (curr_diag - prev_diag) / time_for_diag_change
                        if diag_change_rate > DIAG_RATE_THRESHOLD_NEAR:
                            distance_adjustment_factor = ADJUST_FACTOR_NEAR
                        elif diag_change_rate < DIAG_RATE_THRESHOLD_FAR:
                            distance_adjustment_factor = ADJUST_FACTOR_FAR
                
                speed_mps_final = speed_mps_initial * distance_adjustment_factor
                speed_kmh = speed_mps_final * 3.6
                if speed_kmh < 0: speed_kmh = 0
                if speed_kmh > 200: speed_kmh = 199

        # --- Etiket Güncelleme Mantığı ---
        # `current_time_sec` genel video zamanıdır.
        # `track_last_label_update_time[track_id]` bu araç için son güncelleme zamanı.
        # Eğer bu araç için cache boşsa VEYA yeterli zaman geçtiyse etiketleri güncelle.
        if not track_cached_label_parts[track_id] or \
           (current_time_sec - track_last_label_update_time[track_id] >= LABEL_UPDATE_INTERVAL_SEC):
            
            current_label_parts = [f"ID:{track_id}"]
            if speed_kmh > 0.5:
                current_label_parts.append(f"{int(speed_kmh)}km/h")
            
            # İsteğe bağlı debug bilgileri (saniyede bir güncellenir)
            # if velocity_angle_deg_smoothed is not None:
            #     current_label_parts.append(f"MvAng:{int(velocity_angle_deg_smoothed)}d")
            # current_label_parts.append(f"Phi:{phi_deg_orientation:.0f}d")
            # if width_correction_factor_applied > 1.01 :
            #      current_label_parts.append(f"WCF:{width_correction_factor_applied:.1f}x")

            track_cached_label_parts[track_id] = current_label_parts
            track_last_label_update_time[track_id] = current_time_sec
        
        # --- Görselleştirme (Her zaman önbellekten çiz) ---
        # Sınırlayıcı kutu rengini belirle (bu her frame'de güncel olabilir)
        box_color = (0, 255, 0) 
        if phi_deg_orientation >= MAX_ANGLE_FOR_WIDTH_CORRECTION_DEG :
            box_color = (0, 165, 255) 
        elif width_correction_factor_applied > 1.05 : 
            box_color = (255, 0, 0) 

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Önbellekten alınan etiketleri çizdir
        label_y_start_pos = y1 - 7
        if track_cached_label_parts[track_id]: # Eğer cache'de bir şey varsa
            for i, part_text in enumerate(track_cached_label_parts[track_id]):
                cv2.putText(annotated_frame, part_text, (x1 + 3, label_y_start_pos - (i * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, part_text, (x1 + 3, label_y_start_pos - (i * 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    ids_to_delete_from_history = [tid for tid in track_history_coords.keys() if tid not in prev_tracks]
    for tid_del in ids_to_delete_from_history:
        if tid_del in track_history_coords: del track_history_coords[tid_del]
        if tid_del in track_history_times: del track_history_times[tid_del]
        if tid_del in track_history_diags: del track_history_diags[tid_del]
        if tid_del in track_history_angles: del track_history_angles[tid_del]
        # Yeni eklenen sözlüklerden de sil
        if tid_del in track_last_label_update_time: del track_last_label_update_time[tid_del]
        if tid_del in track_cached_label_parts: del track_cached_label_parts[tid_del]


    elapsed_since_start = time.time() - processing_start_time
    actual_processing_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
    cv2.putText(annotated_frame, f"Processing FPS: {actual_processing_fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Hiz Tahmini (Aci Duzeltmeli v3 - Etiket Seyreltme)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("İşlem tamamlandı.")
