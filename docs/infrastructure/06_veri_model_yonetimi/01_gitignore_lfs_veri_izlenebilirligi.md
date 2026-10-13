# Veri, Model Ağırlıkları ve Depo Hijyeni

Bu rapor, ilk `git init` + `git add .` anından **önce** çözülmesi gereken en
acil konuyu ele alıyor: depoda ~1.3 GB ikili dosya var ve `.gitignore` yok.

---

## 1. Sorunun boyutu (ölçüldü)

```console
$ du -sh "Yapay zeka" fotograflar firmware .
511M    Yapay zeka
303M    fotograflar
 23M    firmware
188M    .            ← kök dizin, alt dizinler hariç
```

Toplam ~1.3 GB. En büyük tek dosyalar:

| Boyut | Dosya | Depoya girebilir mi |
| --- | --- | --- |
| 273 MB | `Yapay zeka/yolov8x.onnx` | **hayır** — GitHub sert sınırı 100 MB |
| 137 MB | `Yapay zeka/yolov8x.pt` | **hayır** — 100 MB sınırı |
| 74 MB | `fotograflar/pciletop.mkv` | hayır (50 MB uyarı eşiği üstü) |
| 53 MB | `best.pt` | hayır |
| 51 MB | `fotograflar/top.mp4` | hayır |
| 51 MB | `fotograflar/top2.mp4` | hayır (üstelik `top.mp4` ile aynı boyut — muhtemelen kopya) |
| 51 MB | `fotograflar/22.mp4` | hayır |
| 23 MB | `yolov8s.pt` (kök) | hayır |
| 23 MB | `Yapay zeka/yolov8s.pt` | hayır (köktekiyle aynı boyut — kopya) |
| 9.1 MB | `firmware/.pio/build/.../firmware.map` | hayır — derleme çıktısı |
| 6.1 MB | `firmware/.pio/build/.../firmware.elf` | hayır — derleme çıktısı |

`top.mp4` ve `top2.mp4` aynı boyutta (50938302 bayt). Kopya olma olasılığı
yüksek; `sha256sum` ile doğrulanmalı:

```bash
sha256sum fotograflar/top.mp4 fotograflar/top2.mp4 "Yapay zeka/yolov8s.pt" yolov8s.pt
```

Aynı hash çıkarsa diskte ~74 MB ve ~23 MB yerden tasarruf edilir.

### Neden bu kadar ciddi

GitHub sınırları:

| Sınır | Değer | Aşılırsa |
| --- | --- | --- |
| Dosya başına uyarı | 50 MB | push başarılı, uyarı |
| Dosya başına sert | 100 MB | **push reddedilir** |
| Depo başına önerilen | 1 GB | performans uyarısı |
| Git LFS ücretsiz kota | 1 GB depolama / 1 GB bant genişliği (ay) | ücretli plana geçiş gerekir |
| Push başına | 2 GB | push reddedilir |

`yolov8x.onnx` (273 MB) tek başına push'u reddettirir. `.gitignore` yazılmadan
`git add .` yapılırsa bu dosya index'e girer ve **push hiç başarılı olmaz**.
Daha kötüsü: başarılı bir push'tan sonra büyük dosyayı geçmişten silmek
`git filter-repo` gerektirir ve tüm commit SHA'ları değişir — yani depo
geçmişi yeniden yazılır, varsa işbirlikçilerin klonları bozulur.

**Sonuç: `.gitignore` ilk commit'ten önce yazılmalı. Bu geri alınamaz bir
işlemin ön koşuludur.**

---

## 2. `.gitignore`

```text
# ===== Python =====
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# ===== Sanal ortamlar =====
.venv/
venv/
env/
ENV/
my_new_venv/
env.bak/
venv.bak/
.python-version

# ===== Araç önbellekleri =====
.mypy_cache/
.dmypy.json
dmypy.json
.pytest_cache/
.ruff_cache/
.coverage
.coverage.*
coverage.xml
htmlcov/
.tox/
.nox/
.cache/
*.coverprofile

# ===== Dağıtım / paketleme =====
uv.lock.bak
*.whl
*.tar.gz

# ===== Makine öğrenmesi ağırlıkları =====
*.pt
*.pth
*.onnx
*.pb
*.h5
*.tflite
*.safetensors
*.ckpt
runs/
wandb/
mlruns/

# ===== Medya ve ham veri =====
*.mp4
*.mkv
*.avi
*.mov
*.webm
*.png
*.jpg
*.jpeg
*.gif
*.bmp
*.tiff
*.pdf

# ===== Loglar ve seri kayıtlar =====
*.log
serial.log
/tmp/

# ===== Firmware / PlatformIO =====
firmware/.pio/
firmware/.vscode/
*.o
*.obj
*.elf
*.map
*.bin
*.hex
*.d

# ===== İşletim sistemi kalıntıları =====
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# ===== Editör =====
.idea/
.vscode/
*.swp
*.swo
*~
.project
.settings/

# ===== Proje içi arşiv ve çalışma alanları =====
archive/
fotograflar/
Yapay zeka/
data/raw/
data/processed/
models/

# ===== Ortam ve gizli bilgiler =====
.env
.env.*
!.env.example
*.pem
*.key
platformio.local.ini
firmware/platformio.local.ini

# ===== Geçici / yanlışlıkla oluşmuş =====
,*
```

### Tasarım kararları

**`*.png` global olarak yok sayılıyor, ama `docs/` içindeki görseller gerekli.**
`README.md` ve makale figürleri için istisna şart:

```text
# docs altındaki görseller depoya girer
!docs/**/*.png
!docs/**/*.jpg
!docs/**/*.jpeg
!docs/**/*.svg
```

`.gitignore`'da `!` ile istisna yazarken sıra önemli: bir üst dizin yok
sayılıyorsa, altındaki dosya için istisna **çalışmaz**. Örnek: `fotograflar/`
yok sayılıyorsa `!fotograflar/onemli.png` işe yaramaz. `docs/` dizini yok
sayılmadığı için `!docs/**/*.png` çalışır.

**`,*` satırı** kök dizindeki `,` adlı yetim dosyayı hedefliyor. `file`
çıktısı bunu Python script'i olarak tanımladı (`00_durum_analizi/01_mevcut_envanter.md`
bölüm 5). Silmek daha doğru, ama `.gitignore` ile de korunaklı hale gelir.

**`*.pt` global.** `yolov8n.pt` ultralytics tarafından otomatik indiriliyor,
depoya girmesi gereksiz. `best.pt` ve `detector.pt` ise GitHub Release'ten
indirilecek (`03_ci_cd/04_release_versiyonlama.md` bölüm 6).

**`platformio.local.ini`.** `02_kod_kalitesi/04_firmware_c_statik_analiz.md`
bölüm 2'de tanımlandığı gibi port ayarları bu dosyada tutulur ve depoya girmez.

---

## 3. Git LFS kararı: kullanılmıyor

| Yöntem | Kota | Sürümleme | Karar |
| --- | --- | --- | --- |
| Git LFS | 1 GB depolama + 1 GB/ay bant | her commit'te yeni kopya | **reddedildi** |
| GitHub Release eki | 2 GB/dosya, toplam kota yok | tag'e bağlı | **kabul** |
| Zenodo | 50 GB | DOI | **kabul** (ham veri) |
| Hugging Face Hub | geniş | model kartı | NRC checkpoint için değerlendirilecek |

LFS'in reddedilme gerekçesi somut: `best.pt` 53 MB ve her yeniden eğitmede
yeni bir sürüm commit'lenirse kota birkaç ayda dolar. Kota aşımında LFS
**sessizce** çalışmayı durdurur — `git push` başarılı görünür ama dosyalar
yüklenmez, klonlayan kişi pointer dosyaları alır. Bu sınıf bir arıza çok geç
fark edilir.

Release eki yaklaşımında her sürümün ağırlıkları tag'e bağlı kalır:

```bash
gh release upload v0.1.0 best.pt detector.pt --clobber
gh release download v0.1.0 --pattern "*.pt" --dir models/
```

Ham video verileri için Release uygun değil (74 MB'lık `pciletop.mkv` gibi
dosyalar sürümlenmiyor, yalnızca arşivleniyor). Bunlar Zenodo'ya.

---

## 4. Bozuk `.git` onarımı

**Bu adım uzak depo etkilemez, yalnızca yereldir.** Yine de geri alınamaz
sonuçları olabilir — `.git` dizini silinirse geçmiş kaybolur. Aşağıdaki sıra
hiçbir şeyi silmez.

### 4.1 Mevcut durumu koru

```bash
cd /home/kadir/Masaüstü/Goruntu-Isleme-Hiz-Tespiti

# .git'i yeniden adlandır (SİLME)
mv .git .git.broken

# içindekileri incele
ls -la .git.broken/
cat .git.broken/config
cat .git.broken/packed-refs
```

`config` ve `packed-refs` içeriği uzak depo adresini gösterebilir. Varsa, bu
bilgi yeni depo kurulumunda kullanılır:

```bash
cat .git.broken/config
# [remote "origin"] url = ... satırına bak
cat .git.broken/packed-refs
# hangi branch/tag'ler vardı
```

`index` dosyası (8851 bayt) daha önce hangi dosyaların izlendiğini gösterir:

```bash
GIT_INDEX_FILE=.git.broken/index git ls-files 2>/dev/null || \
  strings .git.broken/index | head -50
```

Bu çıktı, `.gitignore` yazılırken hangi dosyaların daha önce izlendiğini
gösterir — değerli bir referans.

### 4.2 Kurtarma mümkün mü

```bash
# objects dizini gerçekten boş mu
find .git.broken/objects -type f | wc -l
# beklenen: 0

# pack dosyası var mı
ls -la .git.broken/objects/pack/ 2>/dev/null
```

`objects/` boş ve `pack/` yoksa **kurtarılacak commit yok**. Geçmiş fiilen
kaybolmuş. Bu durumda temiz başlangıç tek seçenek.

Eğer bir `.pack` dosyası varsa:

```bash
git --git-dir=.git.broken verify-pack -v .git.broken/objects/pack/*.idx
git --git-dir=.git.broken cat-file --batch-all-objects --batch-check | head
```

Bu komutlar kurtarılabilir nesneleri listeler. Bu senaryo doğrulanmadı —
`find` çıktısı 0 ise atlanabilir.

### 4.3 Temiz başlangıç

```bash
# 1. .gitignore yaz (bölüm 2) — İLK ADIM, add'den önce
$EDITOR .gitignore

# 2. yeni depo
git init -b main

# 3. ne ekleneceğini ÖNCE gör, ekleme yapmadan
git add --dry-run . 2>&1 | tee /tmp/would-add.txt
wc -l /tmp/would-add.txt

# 4. büyük dosya sızması var mı denetle
git add --dry-run . 2>&1 | sed 's/^add //' | tr -d "'" | while read -r f; do
    [ -f "$f" ] || continue
    size=$(stat -c%s "$f")
    [ "$size" -gt 1048576 ] && printf '%10d  %s\n' "$size" "$f"
done | sort -rn

# beklenen: HİÇBİR çıktı yok. Çıktı varsa .gitignore eksik.

# 5. toplam index boyutunu tahmin et
git add --dry-run . 2>&1 | sed 's/^add //' | tr -d "'" | while read -r f; do
    [ -f "$f" ] && stat -c%s "$f"
done | awk '{s+=$1} END {printf "toplam: %.1f MB (%d dosya)\n", s/1048576, NR}'

# beklenen: < 20 MB

# 6. ancak şimdi gerçekten ekle
git add .
git status --short | head -30
git count-objects -vH

# 7. ilk commit
git commit -m "chore: initialize repository with existing research code"
```

**Adım 3-5 atlanmamalı.** `git add .` çalıştırıldıktan sonra büyük bir dosya
index'e girmişse, `git rm --cached` ile çıkarmak mümkün ama hata kolayca gözden
kaçar. Kuru çalıştırma (`--dry-run`) sıfır maliyetli bir güvenlik ağı.

### 4.4 Uzak depo

```bash
# depo oluştur (public mi private mi — karar 05_lisans_citation.md bölüm 6'ya bağlı)
gh repo create OWNER/vehicle-speed-estimation --private --source=. --remote=origin

# veya mevcut bir depoya bağla
git remote add origin git@github.com:OWNER/REPO.git

# push öncesi son kontrol
git log --oneline
git count-objects -vH          # size-pack değeri makul mü
git ls-files | wc -l           # kaç dosya izleniyor

git push -u origin main
```

**Public/private kararı ertelenmemeli ama aceleye de getirilmemeli.**
`05_yonetisim/05_lisans_citation.md` bölüm 6'daki üniversite fikri mülkiyet
sorusu cevaplanmadan public yapılmamalı. Öneri: önce `private`, sorular
netleşince `public`. Private → public geçişi kayıpsız; public → private
geçişinde fork'lar ve arşivler kalır.

---

## 5. Veri seti ve deney izlenebilirliği

Depoya girmeyen veriler için üç katman:

| Katman | İçerik | Nereye | Erişim |
| --- | --- | --- | --- |
| 1 | Model ağırlıkları (`best.pt`, `detector.pt`) | GitHub Release | `gh release download` |
| 2 | Ham video ve deney kayıtları | Zenodo | DOI |
| 3 | Küçük fixture'lar (`telemetry_sample.jsonl`) | **depo içi** | doğrudan |

Üçüncü katman depoya girer çünkü birkaç KB ve testlerin çalışması için şart.

### Zenodo kaydı

Zenodo, GitHub deposundan doğrudan arşiv üretebilir ve DOI verir:

1. `zenodo.org` → GitHub ile giriş
2. Depoyu "enabled" yap
3. Bir release oluştur → Zenodo otomatik yakalar ve DOI üretir

Zenodo'nun yakaladığı şey **release'teki dosyalar**, yani `.gitignore`'daki
videolar değil. Ham veriyi Zenodo'ya yüklemek için ayrı bir "community upload"
gerekir:

```bash
# yerelde paketle
tar -czf vehicle-speed-dataset-v1.tar.gz \
    fotograflar/top.mp4 fotograflar/pciletop.mkv "Yapay zeka"/[0-9].mp4
sha256sum vehicle-speed-dataset-v1.tar.gz > vehicle-speed-dataset-v1.sha256
```

Zenodo'ya yüklerken `.sha256` dosyasını da ekle. Makalede veri seti atfı:

```text
Yıldız, K. (2026). Vehicle Speed Estimation Dataset v1 [Data set].
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

DOI üretilmeden makaleye veri atfı yazılamaz — bu, `CITATION.cff` içindeki
`preferred-citation` alanının boş bırakılma gerekçesiyle aynı
(`03_ci_cd/04_release_versiyonlama.md` bölüm 4).

### Deney kaydı

Her ölçüm koşusu için depoya girmeyen ama izlenebilir bir kayıt:

`data/experiments/` (`.gitignore`'da) altında:

```text
data/experiments/2026-10-13_hfov85_video1/
├── config.toml          # koşunun tüm parametreleri
├── results.csv          # kare kare çıktı
├── telemetry.jsonl      # eşzamanlı ESP32 kaydı
├── git-sha.txt          # hangi commit'te koşuldu
└── notes.md             # gözlemler
```

`git-sha.txt` kritik: iki ay sonra "bu sonuç hangi kod sürümüyle alındı?"
sorusunun tek cevabı. Üretilmesi:

```bash
git rev-parse HEAD > data/experiments/$RUN_ID/git-sha.txt
```

Bu yapı `scripts/collect_dataset.py` ile otomatikleştirilmeli.
`PLANLANIYOR` kategorisinde; `07_uygulama_yol_haritasi/01_adim_adim_uygulama.md`
Faz 6'da yer alıyor.

---

## 6. Depo boyutu bütçesi

`.gitignore` uygulandıktan sonra depoya girecek dosyalar:

| Kategori | Tahmini boyut |
| --- | --- |
| Python kaynak (`src/`, `tests/`, `scripts/`) | ~200 KB |
| Firmware (`firmware/src`, `platformio.ini`) | ~15 KB |
| Doküman (`docs/`, `README.md`, `CONTRIBUTING.md`, ...) | ~500 KB |
| Yapılandırma (`.github/`, `pyproject.toml`, ...) | ~100 KB |
| Fixture'lar (`tests/fixtures/*.jsonl`) | ~50 KB |
| `docs/paper/figures/` (seçilmiş görseller) | ~2-5 MB |
| **Toplam** | **~3-6 MB** |

1.3 GB'dan ~5 MB'a. Bu, klonlamanın saniyeler sürmesi ve CI checkout'ının
anında tamamlanması demek.

`docs/paper/figures/` için dikkat: `amac.png` 1.68 MB,
`Gemini_Generated_Image_*.png` 6.46 MB. İkincisi depo dışında kalmalı;
`README.md`'de 6 MB'lık bir görsel sayfa yüklenmesini belirgin şekilde
yavaşlatır. Gerekliyse küçültülmeli:

```bash
# görsel boyutunu kontrol et
identify amac.png            # ImageMagick
# küçült (kalite kaybıyla)
convert amac.png -resize 1600x -quality 85 docs/paper/figures/architecture.png
ls -la docs/paper/figures/architecture.png
```

WebP'ye çevirmek daha da etkili:

```bash
cwebp -q 85 amac.png -o docs/paper/figures/architecture.webp
```

Ama `.gitignore`'da `*.png` yok sayıldığı için `!docs/**/*.png` istisnasının
yazılması şart (bölüm 2).

---

## 7. Doğrulama

```bash
# .gitignore kuralları gerçekten eşleşiyor mu
git check-ignore -v best.pt video1.mp4 fotograflar/top.mp4 "Yapay zeka/yolov8x.onnx" \
  firmware/.pio/build/esp32doit-devkit-v1/firmware.elf
# her satır için bir kural göstermeli

# istisnalar çalışıyor mu
git check-ignore -v docs/paper/figures/architecture.png
# ÇIKTI OLMAMALI (yani yok sayılmıyor)

# izlenen dosyalar arasında büyük olan var mı
git ls-files -z | xargs -0 -I{} sh -c 'test -f "{}" && stat -c "%s {}" "{}"' | \
  sort -rn | head -10

# depo hacmi
git count-objects -vH
du -sh .git
```

Son komutun `size-pack` değeri ~5 MB civarında olmalı. 50 MB'ı aşıyorsa
`.gitignore`'da bir eksik var.

---

## 8. Açık noktalar

1. **`top.mp4` / `top2.mp4` kopya mı?** `sha256sum` ile doğrulanmadı. Kopya
   ise diskte 51 MB yer açılır; depoya girmeyecekleri için depo boyutunu
   etkilemez.
2. **`Yapay zeka/` dizini silinmeli mi?** 511 MB, içinde eski denemeler ve bir
   venv var. Silinmesi araştırma geçmişini kaybettirir. Öneri: `archive/`
   altına taşınıp `.gitignore`'da tutulsun, diskte kalsın. Karar kullanıcıya
   ait — bu rapor silme önermiyor.
3. **PDF'ler depoya girmiyor** (`*.pdf` yok sayılıyor). Ama
   `Makaleyeni.pdf` ve `Araç Hız Tespiti Makale Tamamlama_(1).pdf` önceki
   makale taslakları. Depo dışında kişisel arşivde tutulmalı; `docs/paper/`
   altında Markdown sürümleri olmalı. PDF'ler diff'lenemez, bu yüzden depoda
   tutmanın faydası yok.
4. **`kadir_yildiz_2510101209_sunum.pdf` (8.1 MB).** Sunum, makalenin bir
   türevi. Depoya girmez. Zenodo'ya "ek materyal" olarak yüklenebilir.
