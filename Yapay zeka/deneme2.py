import cv2
import numpy as np
import math

def find_sphere(image_path, dp=1.2, min_dist=100, param1=100, param2=30, min_radius=10, max_radius=100):
    """
    Verilen bir görüntüdeki küresel nesneyi (daireyi) bulur.
    Hough Circle Transform parametreleri, görüntünüze göre ayarlanmalıdır.

    Args:
        image_path (str): Görüntü dosyasının yolu.
        dp (float): Çözünürlük oranı.
        min_dist (int): Tespit edilen dairelerin merkezleri arasındaki minimum mesafe.
        param1 (int): Canny kenar tespiti için üst eşik değeri.
        param2 (int): Daire merkezlerini tespit etmek için eşik değeri.
        min_radius (int): Tespit edilecek minimum daire yarıçapı (piksel).
        max_radius (int): Tespit edilecek maksimum daire yarıçapı (piksel).

    Returns:
        (x, y, r) (tuple): Dairenin merkez koordinatları (x, y) ve yarıçapı (r).
                           Eğer daire bulunamazsa None döner.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Hata: Görüntü yüklenemedi -> {image_path}")
        return None

    # Gürültüyü azaltmak için görüntüyü gri tonlamalıya çevir ve bulanıklaştır
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circle Transform ile daireleri bul
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        # Tespit edilen ilk daireyi al
        circles = np.uint16(np.around(circles))
        first_circle = circles[0, 0]
        x, y, r = first_circle[0], first_circle[1], first_circle[2]
        return (x, y, r)
    else:
        print(f"Uyarı: '{image_path}' dosyasında küresel nesne bulunamadı.")
        return None

def calculate_3d_displacement(ref_image_path, new_image_path, real_diameter, ref_distance):
    """
    İki görüntü arasındaki küresel nesnenin 3D yer değiştirmesini hesaplar.

    Args:
        ref_image_path (str): Referans görüntünün dosya yolu.
        new_image_path (str): Yeni görüntünün dosya yolu.
        real_diameter (float): Kürenin gerçek çapı (kullandığınız birimde, örn: mm).
        ref_distance (float): Kameranın küreye olan mesafesi (referans fotoğrafta).

    Returns:
        dict: Hesaplanan yer değiştirme bilgilerini içeren bir sözlük.
              Eğer bir hata olursa None döner.
    """
    # 1. Adım: Her iki görüntüde de kürenin konumunu ve piksel boyutunu bul
    print("Referans görüntü işleniyor...")
    ref_sphere = find_sphere(ref_image_path)
    if ref_sphere is None:
        return None

    print("Yeni görüntü işleniyor...")
    new_sphere = find_sphere(new_image_path)
    if new_sphere is None:
        return None

    ref_x_px, ref_y_px, ref_r_px = ref_sphere
    new_x_px, new_y_px, new_r_px = new_sphere

    # Referans görüntüdeki piksel cinsinden çap
    ref_diameter_px = 2 * ref_r_px

    # 2. Adım: Kameranın piksel cinsinden odak uzaklığını hesapla (Kalibrasyon)
    # Bu formül, iğne deliği kamera modelinden gelir:
    # F_px = (D_px * Z_real) / D_real
    focal_length_px = (ref_diameter_px * ref_distance) / real_diameter
    print(f"\nHesaplanan Odak Uzaklığı: {focal_length_px:.2f} piksel")

    # 3. Adım: Yeni fotoğraftaki nesnenin kameraya olan yeni uzaklığını hesapla
    # Z_new_real = (F_px * D_real) / D_new_px
    new_distance = (focal_length_px * real_diameter) / (2 * new_r_px)
    print(f"Nesnenin Yeni Uzaklığı: {new_distance:.2f} birim")


    # 4. Adım: Piksel cinsinden yer değiştirmeyi hesapla
    delta_x_px = new_x_px - ref_x_px
    delta_y_px = new_y_px - ref_y_px

    # 5. Adım: Piksel yer değiştirmesini gerçek dünyadaki yer değiştirmeye çevir
    # Bu formül de iğne deliği kamera modelinin bir sonucudur:
    # X_real = (X_px * Z_real) / F_px
    delta_x_real = (delta_x_px * new_distance) / focal_length_px
    delta_y_real = (delta_y_px * new_distance) / focal_length_px # y ekseni genellikle ters olduğu için - olabilir
    
    # 6. Adım: Z eksenindeki (derinlik) yer değiştirmeyi hesapla
    delta_z_real = new_distance - ref_distance

    # 7. Adım: Toplam 3D yer değiştirmeyi (Öklid mesafesi) hesapla
    total_displacement = math.sqrt(delta_x_real**2 + delta_y_real**2 + delta_z_real**2)

    # Sonuçları bir sözlükte topla
    results = {
        "referans_konum_px": (ref_x_px, ref_y_px),
        "yeni_konum_px": (new_x_px, new_y_px),
        "referans_yarıçap_px": ref_r_px,
        "yeni_yarıçap_px": new_r_px,
        "referans_mesafe_birim": ref_distance,
        "yeni_mesafe_birim": new_distance,
        "yer_değiştirme_x_birim": delta_x_real,
        "yer_değiştirme_y_birim": delta_y_real,
        "yer_değiştirme_z_birim": delta_z_real,
        "toplam_3d_yer_değiştirme_birim": total_displacement
    }
    
    return results

if __name__ == '__main__':
    # --- KULLANICI GİRDİLERİ ---
    # Bu değerleri kendi durumunuza göre güncelleyin.

    # 1. Görüntü Dosyalarının Yolları
    REFERANS_GORUNTU_YOLU = 'referans.png'  # Kürenin ilk konumunun fotoğrafı
    YENI_GORUNTU_YOLU = 'yenikonum.png'    # Kürenin yer değiştirdiği fotoğraf

    # 2. Kürenin Gerçek Dünya Bilgileri
    KURE_GERCEK_CAPI = 190.0  # Örneğin milimetre (mm) cinsinden
    KAMERADAN_REFERANS_UZAKLIK = 1370.0 # Referans fotoğrafta kameranın küreye olan uzaklığı (mm)

    # --- HESAPLAMA ---
    displacement_data = calculate_3d_displacement(
        ref_image_path=REFERANS_GORUNTU_YOLU,
        new_image_path=YENI_GORUNTU_YOLU,
        real_diameter=KURE_GERCEK_CAPI,
        ref_distance=KAMERADAN_REFERANS_UZAKLIK
    )

    # --- SONUÇLARI GÖSTERME ---
    if displacement_data:
        print("\n--- YER DEĞİŞTİRME ANALİZ SONUÇLARI ---")
        print(f"Kürenin gerçek çapı: {KURE_GERCEK_CAPI} birim")
        print("-" * 40)
        print(f"Referans Görüntü:")
        print(f"  > Konum (piksel): {displacement_data['referans_konum_px']}")
        print(f"  > Yarıçap (piksel): {displacement_data['referans_yarıçap_px']}")
        print(f"  > Mesafe (birim): {displacement_data['referans_mesafe_birim']:.2f}")
        print("-" * 40)
        print(f"Yeni Görüntü:")
        print(f"  > Konum (piksel): {displacement_data['yeni_konum_px']}")
        print(f"  > Yarıçap (piksel): {displacement_data['yeni_yarıçap_px']}")
        print(f"  > Hesaplanan Mesafe (birim): {displacement_data['yeni_mesafe_birim']:.2f}")
        print("-" * 40)
        print("Gerçek Dünyadaki Yer Değiştirme (Deplasman):")
        print(f"  > X Ekseni: {displacement_data['yer_değiştirme_x_birim']:.2f} birim")
        print(f"  > Y Ekseni: {displacement_data['yer_değiştirme_y_birim']:.2f} birim")
        print(f"  > Z Ekseni (Derinlik): {displacement_data['yer_değiştirme_z_birim']:.2f} birim")
        print("\n" + "=" * 40)
        print(f"  TOPLAM 3D YER DEĞİŞTİRME: {displacement_data['toplam_3d_yer_değiştirme_birim']:.2f} birim")
        print("=" * 40)
