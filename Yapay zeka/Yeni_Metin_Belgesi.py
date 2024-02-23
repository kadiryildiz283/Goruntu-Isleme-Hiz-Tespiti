import cv2
import numpy as np
import math
from math import dist

# YOLOv4-tiny modelini yükle
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

cap = cv2.VideoCapture("3.mp4")

# Önceki karedeki nesne bilgilerini saklamak için sözlük
prev_cx = {}
prev_cy = {}
prev_time = {}
yenicaprazalt = 0
eskix,eskiy=0,0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1 and class_id == classes.index("orange"):
                center_x, center_y, w, h = (np.array(detection[0:4]) * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cx, cy = int(x + w / 2), int(y + h / 2)

            # Şu anki karedeki zamanı al
            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # saniye cinsinden

            # Eğer nesne daha önce görüldüyse
            if i in prev_cx and i in prev_cy and i in prev_time:
                # Önceki karedeki cx, cy ve zaman değerlerini al
                px = prev_cx[i]
                py = prev_cy[i]
                pt = prev_time[i]

                # Nesnenin hareket ettiği mesafeyi hesapla
                distance = dist((px, py), (cx, cy))  # piksel cinsinden

                # Kare aralığındaki zaman farkını hesapla
                elapsed_time = curr_time - pt  # saniye cinsinden
                caprazalt = math.sqrt(((x)*(y+h)+(x+w)*(y)))
                if caprazalt < yenicaprazalt:
                    print("uzaklaşıyor")
                elif caprazalt > yenicaprazalt and yenicaprazalt != 0:
                    print("yakınlaşıyor")
                else:
                    print("sabit")
                    # gidiş yönünün açısını hesapla
                x_bileseni = cx
                y_bileseni = cy

                # atan2 fonksiyonunu kullanarak açıyı hesapla
                aci = math.atan2(y_bileseni, x_bileseni)

                # açıyı dereceye çevir
                aci = math.degrees(aci)
                if x_bileseni >= eskix and y_bileseni <= eskiy:
                    # cisim sağ üst bölgede
                    eskix,eskiy = x_bileseni,y_bileseni
                    aci = aci
                    print("sağ üste gidiyor")
                elif x_bileseni < eskix and y_bileseni <= eskiy:
                    # cisim sol üst bölgede
                    eskix,eskiy = x_bileseni,y_bileseni
                    aci = aci + 90
                    print("sol üste gidiyor")
                elif x_bileseni < eskix and y_bileseni > eskiy:
                    # cisim sol alt bölgede
                    aci = aci + 180
                    print("sol alta gidiyor")
                    eskix,eskiy = x_bileseni,y_bileseni
                else:
                    # cisim sağ alt bölgede
                    aci = aci + 270
                    print("sağ alta gidiyor")
                    eskix,eskiy = x_bileseni,y_bileseni
                print("gidiş yönünün açısı:", aci, "derece")
                yenicaprazalt = caprazalt;
                try:
                    speed_ms = distance / elapsed_time  # bölme işlemi yap
                    speed_kh = speed_ms / 50
                except ZeroDivisionError:
                    speed_kh = 0
                    print("Zaman farkı sıfır olduğu için hız hesaplanamadı.")

                # Hızı ekranda göster
                cv2.putText(frame, f"Hız: {int(speed_kh)} m/s", (x, y - 50), font, 0.5, (255, 255, 255), 2)

            # Şu anki karedeki cx, cy ve zaman değerlerini sakla
            prev_cx[i] = cx
            prev_cy[i] = cy
            prev_time[i] = curr_time

            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.line(frame, (x,y), (x+w, y+h), (255, 255, 255), 5)
            cv2.rectangle(frame, (x, y), (x +w , y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y - 5), font, 0.5, color, 2)

    cv2.imshow("YOLOv4-tiny - Portakal Algılama ve Hız Ölçümü", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
