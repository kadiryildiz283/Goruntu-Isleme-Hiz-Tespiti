# Proje Değerlendirmesi ve SCI-E Q2 Makale Fizibilite Notu (Hakem Gözüyle Analiz)

**Tarih:** 14 Ekim 2026  
**Kapsam:** `src/vehicle_speed/`, `firmware/`, `scripts/` ve geçmiş iterasyonların (`eski/`) uçtan uca incelenmesi sonucunda hazırlanan teknik ve akademik değerlendirme raporu.

---

## 1. Genel Karar: Q2 Seviyesinde Makale Çıkar mı?

**Evet, bu projeden kesinlikle SCI-E Q2 (hatta kurgu ve deneyler eksiksiz tamamlanırsa *Elsevier Measurement* gibi Q1) seviyesinde bir makale çıkar.**

Ancak makalenin "şu anki haliyle hemen gönderilmesi" durumunda değil; eldeki çok güçlü yazılım/mekatronik iskeletin üzerindeki **4 kritik akademik ve deneysel eksiğin** tamamlanmasıyla kabul alacaktır.

---

## 2. Projenin En Güçlü 3 Yönü (Q2'ye Taşıyacak Silahlar)

### 2.1. Mekatronik "Ground Truth" (Yer Gerçeği) Düzeneği (En Büyük Fark)
Monoküler hız tespiti makalelerinin %80'i hakemlerden şu soru nedeniyle ret alır:
> *"Kamerayla hızı ölçtüğünü iddia ediyorsun ama aracın gerçek hızını (ground truth) neyle ölçtün? Elindeki referansın ölçüm belirsizliği nedir?"*

Bu projedeki **ESP32 + TMC2208 + T8 vidalı mil (400 adım/mm = 2.5 µm/adım)** düzeneği (`firmware/src/main.cpp` ve `src/vehicle_speed/telemetry/ground_truth.py`), *Hardware-in-the-Loop (HIL)* ve metroloji odaklı dergilerin (*Measurement*, *MDPI Sensors*, *IEEE Access*) tam olarak aradığı fiziksel kanıttır.

### 2.2. Fizik Destekli Hibrit Mimari (Pinhole + 3D Kalman + NRC)
Yalnızca YOLO + Pinhole + Kalman kullanan bir çalışma güncel literatürde **"Incremental / Novelty Yok"** gerekçesiyle masa başı reddi (desk reject) alır. Ancak:
- Pinhole modelindeki $\sigma_Z \propto Z^2 \cdot \sigma_w$ karesel hata patlamasını matematiksel olarak teşhis etmek,
- Bunun üzerine **5 fiziksel öznitelikli (`NRCFeatures`: `v_raw_ms`, `pixel_width`, `y_normalized`, `aspect_ratio`, `confidence`) Nöral Artık Kalibratör (`NeuralResidualCalibrator`)** eklemek,
- Doğrudan hız yerine klasik fizik modelinin yaptığı artık hatayı ($\Delta v = v_{\text{gt}} - v_{\text{raw}}$) öğrenmek (*Physics-Informed Residual Learning*),

makaleye doğrudan **metodolojik özgünlük (novelty)** kazandırmaktadır.

### 2.3. Yazılım Mühendisliği Kalitesi ve Tekrarlanabilirlik (Reproducibility)
Akademik projelerin kod tabanı genellikle tek dosyada spagetti koddan oluşur. Bu projede `src/vehicle_speed/` altında:
- Sıfır harici bağımlılıklı saf optik geometri (`src/vehicle_speed/geometry.py`),
- Joseph stabilite formlu 6 durumlu 3B Kalman filtresi (`src/vehicle_speed/filters/kalman.py`),
- Modüler boru hattı (`src/vehicle_speed/pipeline.py`),
- `mypy --strict`, `ruff` ve `%85+` birim/entegrasyon test kapsamı (`tests/`)

bulunması, makalede *"Code, hardware firmware, and dataset are publicly available and reproducible"* vurgusuyla hakemlerden tam puan alınmasını sağlar.

---

## 3. Hakem Gözüyle Kritik Eksikler (Gönderim Öncesi Tamamlanması Şart Olanlar)

Bir Q2 dergi hakemi kodu ve sistemi incelediğinde aşağıdaki **4 kritik noktayı** sorgulayacaktır:

### ⚠️ Kritik Nokta 1: Firmware'de Telemetri Örnekleme Hızı (5 Saniye Darboğazı)
* **Mevcut Durum:** `firmware/src/main.cpp` (satır 105) içinde periyodik durum logu `if (millis() - lastLogTime > 5000)` olarak ayarlanmış (5 saniyede bir konum raporu).
* **Sorun:** 500 mm'lik vidalı milde vagon zaten birkaç saniye içinde uçtan uca gidip yön değiştirmektedir. 5 saniyede bir alınan `status` olayıyla `scripts/collect_dataset.py` içindeki `interpolate_ground_truth_speed` fonksiyonu ivmelenme (ramp-up/ramp-down) ve yön değişimlerini kaçırır, hızı olduğundan düşük ve hatalı enterpole eder.
* **Çözüm:**
  1. Seri port loglama periyodu **20 ms – 50 ms (20–50 Hz)** seviyesine indirilmelidir.
  2. `direction_change` olayları da yörünge/hız enterpolasyonuna dahil edilmelidir.
  3. `stepper.run()` döngüsünü bloklamamak için seri port baud hızı gerekirse `230400` yapılmalı ve JSON çıktısı kısa tutulmalıdır.

### ⚠️ Kritik Nokta 2: NRC Eğitim, Normalizasyon ve Ablasyon Betikleri (`train_nrc.py`)
* **Mevcut Durum:** `src/vehicle_speed/nrc/model.py` içinde NumPy tabanlı ileri besleme (inference) ve `.npz` ağırlık kaydetme/yükleme hazır; `scripts/collect_dataset.py` ile veri toplama aracı da yazılmış. Ancak toplanan `.npz` veri setiyle bu ağı eğitecek eğitim betiği henüz yok.
* **Sorun:** `NRCFeatures` içinde `pixel_width` ($10\text{–}800\text{ px}$) ile `confidence` ($0.0\text{–}1.0$) aynı vektörde ham olarak modele giriyor. Normalizasyon yapılmazsa büyük ölçekli özellik (`pixel_width`) gradyanları domine eder.
* **Çözüm:**
  1. Girdi öznitelikleri için **Z-Score (`mean`, `std`) normalizasyonu** `NeuralResidualCalibrator` içine ve `.npz` ağırlık dosyasına eklenmelidir.
  2. PyTorch/NumPy ile Huber Loss (Smooth L1) kullanan, K-Fold Cross Validation yapan ve makaledeki **Ablasyon Tablosunu** (1 girdi, 2 girdi, ..., 5 girdi; Baseline vs. Kalman vs. Kalman+NRC) otomatik üreten `scripts/train_nrc.py` ve `scripts/evaluate_benchmark.py` betikleri yazılmalıdır.

### ⚠️ Kritik Nokta 3: Donanım Arızasının Giderilmesi ve Gerçek Deney Matrisi
* **Mevcut Durum:** `eski/DONANIM_ARIZA_VE_TEHIS_RAPORU.md` içinde teşhis edildiği üzere ESP32 kartındaki dahili kısa devre (internal latch-up) nedeniyle yeni ESP32 ile fiziksel deney kayıtlarının alınması gerekmektedir.
* **Çözüm:** Q2 makale için simülasyon veya mock veri tek başına yetmez. Yeni ESP32 devreye alındıktan sonra:
  - En az **5–8 farklı step hız profilinde** ($1000\text{–}8000\text{ adım/s}$),
  - **3–4 farklı kamera mesafesi ve bakış açısında** ($50\text{ cm}, 100\text{ cm}, 150\text{ cm}, 200\text{ cm}$ ve açılı çekimler)
  gerçek vidalı mil + kamera senkronize veri seti toplanmalıdır.

### ⚠️ Kritik Nokta 4: "Ölçek Sorunu" — Laboratuvar Kızağından Gerçek Trafiğe Köprü (Scale Invariance)
* **Hakem Sorusu:** *"Vidalı milde hızlar $2.5\text{–}20\text{ mm/s}$ ve mesafe $0.5\text{–}2\text{ m}$. Oysa gerçek araçlar $30\text{–}120\text{ km/s}$ hızla ve $10\text{–}60\text{ m}$ uzakta gidiyor. Vidalı milde doğruladığınız NRC modeli gerçek trafikte nasıl genellenecek?"*
* **Makaledeki Savunma Stratejisi:**
  1. Makalenin ana odağını **"Sub-Millimeter Hardware-in-the-Loop Benchmarking and Neural Residual Calibration of Monocular 3D Speed Estimation"** (Monoküler hız tahmininde piksel kuantizasyon hatasının mekatronik düzenekle modellenmesi ve NRC ile kalibrasyonu) olarak konumlandırmak.
  2. Hızı ve derinliği **boyutsuz (dimensionless / scale-invariant)** büyüklüklerle (örneğin saniyedeki bağıl derinlik değişimi $\frac{\Delta Z / Z}{\Delta t}$ veya bağıl piksel genişliği değişim oranı $\frac{\Delta w_{px}}{w_{px} \cdot \Delta t}$) ilişkilendirmek. Böylece 1 metredeki kalibrasyon hedefinin optik/kuantizasyon fiziği ile 30 metredeki aracın optik fiziği matematiksel olarak birebir eşdeğer (izomorfik) olur. Bu teorik köprü makaleyi Q2'den Q1 seviyesine dahi taşır.

---

## 4. Özet Durum Tablosu ve Eylem Planı

| Bileşen / Kriter | Mevcut Durum | Q2 Makale İçin Yeterlilik | Yapılacak İşlem |
| :--- | :---: | :--- | :--- |
| **Fikir ve Özgünlük (Hibrit Pinhole + KF + NRC)** | ✅ Hazır | **Çok Yüksek** | Makalede $Z^2$ hata yayılımı ve boyutsuz ölçek (scale-invariance) teorisiyle sunulacak. |
| **Yazılım Mimarisi & Test Altyapısı (`src/`, `tests/`)** | ✅ %85+ Coverage | **Çok Yüksek** | Mevcut yapı korunacak. |
| **ESP32 Firmware & Telemetri (`firmware/src/main.cpp`)** | ⚠️ 5 sn log sıklığı | **Güncellenmeli** | Log aralığı `5000 ms` -> `25-50 ms` yapılacak; yön değişim konumları eklenecek. |
| **NRC Eğitim, Normalizasyon ve Ablasyon Betikleri** | ❌ Henüz Yok | **Zorunlu** | `scripts/train_nrc.py`, öznitelik normalizasyonu (`mean`/`std`) ve ablasyon raporu üreticisi yazılacak. |
| **Fiziksel Deney Çekimleri (Yeni ESP32 ile)** | ⏳ Bekliyor | **Zorunlu** | Yeni ESP32 ile farklı mesafe/hızlarda gerçek video + JSONL telemetri toplanacak. |
