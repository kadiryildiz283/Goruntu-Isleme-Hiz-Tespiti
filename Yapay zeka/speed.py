import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import math
from math import dist

model=YOLO('yolov8s.pt')
cap=cv2.VideoCapture('8.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)

count=0

tracker=Tracker()

#cy1=280
#cy2=170

offset=6
eskix,eskiy=0,0
vh_down={}
counter=[]


vh_up={}
counter1=[]

# eklediğim değişkenler
prev_cx = {} # her araç için önceki karedeki cx değerini saklamak için
prev_cy = {} # her araç için önceki karedeki cy değerini saklamak için
prev_time = {} # her araç için önceki karedeki zamanı saklamak için

while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
   # print(results)
    a=results[0].boxes.data.cpu()
    #print(pd.DataFrame(a.numpy()).astype("float"))
    px=pd.DataFrame(a.numpy()).astype("float")
#    print(px)
    list=[]

    for index,row in px.iterrows():
#        print(row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 # saniye cinsinden

        # eğer araç daha önce görüldüyse
        if id in prev_cx and id in prev_cy and id in prev_time:
            # önceki karedeki cx, cy ve Zaman değerlerini al
            px = prev_cx[id]
            py = prev_cy[id]
            pt = prev_time[id]
                       
            # hızı hesapla
            try: # bu blokta hata oluşabilir
            # aracın hareket ettiği mesafeyi hesapla
               # Açı hesapı doğru bir şekilde çalışıyor: 
               distance = math.sqrt((cy-py)*(cy-py) + (cx-px)*(cx-px))  
               bdeg = (cx-px)
               adeg = (cy-py)
               cos_A = (bdeg**2 + distance**2 - adeg**2) / (2 * bdeg * distance)
               cos_B = (adeg**2 + distance**2 - bdeg**2) / (2 * adeg * distance)
               cos_C = (adeg**2 + bdeg**2 - distance**2) / (2 * adeg * bdeg)
               A = math.degrees(math.acos(cos_A))
               B = math.degrees(math.acos(cos_B))
               C = math.degrees(math.acos(cos_C))
               if A > 90:
                   A = A-90
               if B > 90:
                   B=  B-9
               #Açı hesabı bitmiştir.
               #Aşağısı Gerçek metre bulmaaya çalışmaktadir ama doğru çalışmamaktadir.
               gercektekimesafe = 2/(x4 - x3)
               gercektekimesafe2 = 1.4/(y4-y3)
               elapsed_time = curr_time - pt # saniye cinsinden
               pxhiz = distance * elapsed_time
               reelhiz = math.sqrt((pxhiz/gercektekimesafe))
               print(id, cx,cy,px,py)
               a = input("sadsda")
               # hızı kilometre/saat cinsine çevir
            except ZeroDivisionError: # eğer ZeroDivisionError hatası oluşursa
                reelhiz = 0 # hızı sıfır olarak belirle
                print("Zaman farkı sıfır olduğu için hız hesaplanamadı.") # bir mesaj yazdır


            # hızı ekranda göster
            cv2.putText(frame,str(int(reelhiz))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
            caprazalt = math.sqrt(((x1*x1)+(y1*y1)))
            #print(caprazalt)
        # mevcut karedeki cx, cy ve zaman değerlerini sakla
        prev_cx[id] = cx
        prev_cy[id] = cy
        prev_time[id] = curr_time

    cv2.imshow("RGB", frame)
    

    if cv2.waitKey(int(1000/cap.get(cv2.CAP_PROP_FPS)))&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
