# Issue Forms ve PR Şablonları

Serbest Markdown şablonlar (`.md`) yerine **GitHub Issue Forms** (`.yml`)
kullanılır. Fark: form alanları zorunlu tutulabilir, dropdown ile sabit seçenek
sunulabilir, ve eksik doldurulan issue gönderilemez.

Somut gerekçe: `00_durum_analizi/02_teknik_borc_ve_hatalar.md` A1 maddesindeki
hata sınıfı (atanmamış değişken) yalnızca **tam yığın izi** varsa ayıklanabilir.
Serbest metin şablonda "hata alıyorum" yazan bir bildirim işe yaramaz; form
alanı zorunlu tutulursa yığın izi olmadan issue açılamaz.

---

## 1. `.github/ISSUE_TEMPLATE/config.yml`

Boş issue'ları kapatır ve yönlendirme sağlar:

```yaml
blank_issues_enabled: false
contact_links:
  - name: Kullanım sorusu
    url: https://github.com/OWNER/REPO/discussions
    about: "Nasıl yapılır soruları Discussions'ta sorulmalı, issue değil"
  - name: Güvenlik açığı bildirimi
    url: https://github.com/OWNER/REPO/security/advisories/new
    about: "Güvenlik açıklarını public issue olarak AÇMAYIN"
  - name: Donanım kurulumu
    url: https://github.com/OWNER/REPO/blob/main/docs/hardware_setup.md
    about: "ESP32 + TMC2208 vidalı mil bağlantı rehberi"
```

`blank_issues_enabled: false` önemli. Açık bırakılırsa "New issue" ekranında
"Open a blank issue" seçeneği çıkar ve tüm form yapısı atlanır.

Güvenlik yönlendirmesi için depoda **Private Vulnerability Reporting**
etkinleştirilmeli: Settings → Security → Private vulnerability reporting →
Enable. Bu ayar YAML ile yapılamaz, web arayüzünden yapılır.

---

## 2. `.github/ISSUE_TEMPLATE/bug_report.yml`

```yaml
name: Hata bildirimi
description: Beklenmeyen bir davranış bildirin
title: "[BUG] "
labels: ["type/bug", "status/triage"]
body:
  - type: markdown
    attributes:
      value: |
        Bildirimden önce:
        - [Aranmış issue'lar](../labels/type%2Fbug) içinde aynı sorun var mı bakın.
        - `uv run ruff check .` ve `uv run pytest tests/ -q` çıktısını ekleyin.

  - type: textarea
    id: what-happened
    attributes:
      label: Ne oldu?
      description: Gözlemlediğiniz davranışı somut olarak yazın.
      placeholder: "video1.mp4 üzerinde 30. karede hız 0.0 cm/s olarak raporlanıyor, beklenen ~850 cm/s."
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Ne bekliyordunuz?
      placeholder: "Hızın fiziksel olarak makul bir değer almasını bekliyordum."
    validations:
      required: true

  - type: textarea
    id: reproduction
    attributes:
      label: Yeniden üretme adımları
      description: |
        Başkasının aynı hatayı görebilmesi için gereken tam adımlar.
        Kod bloğu kullanın.
      render: bash
      placeholder: |
        uv sync --extra dev --extra ml
        uv run python -m vehicle_speed.cli --video video1.mp4 --initial-dist 183
    validations:
      required: true

  - type: textarea
    id: traceback
    attributes:
      label: Tam yığın izi / hata çıktısı
      description: Kırpılmamış çıktı. Uzunsa `<details>` içine alın.
      render: text
      placeholder: |
        Traceback (most recent call last):
          File ".../detector.py", line 142, in process_frame
            ...
        UnboundLocalError: cannot access local variable 'track_id'
    validations:
      required: true

  - type: dropdown
    id: component
    attributes:
      label: Hangi bileşen?
      description: Emin değilseniz en yakın olanı seçin, triage'da düzeltilir.
      multiple: true
      options:
        - vision (dedektör / geometri)
        - calibration (HFOV, ölçek referansı)
        - tracking (hız türetimi, filtre)
        - telemetry (seri port, ESP32 veri akışı)
        - firmware (ESP32 / PlatformIO)
        - nrc (nöral kalibratör)
        - ci-cd (workflow, linter, test)
        - packaging (kurulum, bağımlılık)
        - docs
        - data (veri seti, fixture)
        - Bilmiyorum
    validations:
      required: true

  - type: dropdown
    id: severity
    attributes:
      label: Etki
      options:
        - "Sistem sonuç üretmiyor"
        - "Sonuçlar yanlış ama sistem çalışıyor"
        - "Yalnızca belirli girdilerde"
        - "Kozmetik / log mesajı"
    validations:
      required: true

  - type: input
    id: version
    attributes:
      label: Sürüm
      description: "`git describe --tags` veya `uv run python -c \"import vehicle_speed; print(vehicle_speed.__version__)\"`"
      placeholder: "v0.1.0 veya commit SHA"
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Ortam
      render: bash
      placeholder: |
        uv run python -c "import sys, platform; print(sys.version); print(platform.platform())"
        uv pip list | grep -E "torch|ultralytics|opencv|numpy|scipy|pyserial"
    validations:
      required: true

  - type: checkboxes
    id: hardware
    attributes:
      label: Donanım bağlantısı var mı?
      options:
        - label: Fiziksel ESP32 bağlı
        - label: Kamera bağlı
        - label: Yalnızca video dosyası ile test ettim
        - label: Donanımsız, yalnızca birim testlerde gördüm

  - type: checkboxes
    id: checks
    attributes:
      label: Bildirimden önce çalıştırdım
      options:
        - label: "`uv run ruff check .` çıktısını ekledim"
          required: true
        - label: "`uv run pytest tests/ -q` çıktısını ekledim"
          required: true
        - label: En güncel `main` dalında denedim
          required: true
```

### Tasarım kararları

**`component` dropdown'u `multiple: true`.** Bir hata hem `vision` hem
`telemetry`'yi kesebilir (ör. zaman senkronizasyonu). Tek seçim zorlanırsa
bildiren yanlış seçer.

**`traceback` zorunlu.** Serbest metin şablonlarda en sık atlanan alan bu.
Zorunlu tutulunca, triage'da "log'u ekleyebilir misiniz?" turu ortadan kalkar.

**`checks` içindeki `required: true`.** Üç kutunun işaretlenmesi zorunlu.
Bunlar işaretlenmeden issue gönderilemez. Katkı verenler için sürtünme, ama
bakım yükünü belirgin şekilde azaltır.

**`title: "[BUG] "`** ön eki otomatik gelir. `01_label_taksonomisi.md` içindeki
issue başlık sözleşmesiyle uyumlu.

---

## 3. `.github/ISSUE_TEMPLATE/feature_request.yml`

```yaml
name: Özellik isteği
description: Yeni bir yetenek önerin
title: "[FEATURE] "
labels: ["type/feature", "status/triage"]
body:
  - type: textarea
    id: problem
    attributes:
      label: Hangi problemi çözüyor?
      description: |
        "Şu özellik olmalı" yerine "şu işi yapamıyorum çünkü..." biçiminde yazın.
      placeholder: |
        Farklı HFOV değerleriyle deney yaparken her seferinde 1_donem_kadir.py
        içindeki hfov=85.0 satırını elle düzenlemem gerekiyor.
    validations:
      required: true

  - type: textarea
    id: proposal
    attributes:
      label: Önerilen çözüm
      render: python
      placeholder: |
        # CLI bayrağı
        uv run python -m vehicle_speed.cli --hfov 72.5 --video video1.mp4
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Düşündüğünüz alternatifler
      placeholder: "Yapılandırma dosyası, ortam değişkeni, ..."

  - type: dropdown
    id: component
    attributes:
      label: Hangi bileşen?
      multiple: true
      options:
        - vision
        - calibration
        - tracking
        - telemetry
        - firmware
        - nrc
        - ci-cd
        - packaging
        - docs
        - data
    validations:
      required: true

  - type: checkboxes
    id: scope
    attributes:
      label: Kapsam kontrolü
      options:
        - label: Bu değişiklik mevcut API'yi kırmıyor
        - label: Bu değişiklik makale bulgularını etkiliyor (ablation/ölçüm gerekli)
        - label: Fiziksel donanım değişikliği gerektiriyor

  - type: textarea
    id: validation
    attributes:
      label: Doğrulama nasıl yapılacak?
      description: |
        Özelliğin çalıştığı nasıl kanıtlanacak? Hangi test eklenecek?
      placeholder: |
        tests/unit/test_calibration.py içine HFOV'dan odak uzaklığı türetme testi
        eklenir; farklı HFOV değerleri için beklenen f_px değerleri parametrize edilir.
    validations:
      required: true
```

**`validation` alanı zorunlu.** Test edilebilirlik baştan düşünülmezse, özellik
eklenir ama doğrulanamaz. Bu alan, PR şablonundaki "Test kanıtı" bölümüyle
eşleşir.

**`scope` içindeki "makale bulgularını etkiliyor" kutusu** bu depoya özgü.
`00_durum_analizi/01_mevcut_envanter.md` içinde görüldüğü gibi proje bir makale
üretiyor; bir özellik değişikliği ölçüm sonuçlarını değiştiriyorsa ablation
çalışmasının yeniden koşulması gerekir. Bu maliyet baştan görünür olmalı.

---

## 4. `.github/ISSUE_TEMPLATE/research_proposal.yml`

Bu şablon endüstriyel depolarda yok. Gerekçe: proje bir yüksek lisans araştırması
ve algoritmik kararların (Kalman modeli seçimi, NRC mimarisi, ablation tasarımı)
kod değişikliğinden önce tartışılması gerekiyor.

```yaml
name: Araştırma / deney önerisi
description: Algoritma, model mimarisi veya deneysel metodoloji önerisi
title: "[RESEARCH] "
labels: ["type/research", "academic-discussion", "status/triage"]
body:
  - type: markdown
    attributes:
      value: |
        Bu şablon **kod değişikliği** için değil, metodoloji tartışması içindir.
        Somut bir özellik istiyorsanız "Özellik isteği" şablonunu kullanın.

  - type: textarea
    id: hypothesis
    attributes:
      label: Hipotez
      description: Ölçülebilir, yanlışlanabilir bir cümle.
      placeholder: |
        5 girdili artık-hata kalibratörü (NRC), uzak bölgede (Z > 25 m) hız
        RMSE'sini saf Kalman'a göre en az %30 azaltır.
    validations:
      required: true

  - type: textarea
    id: baseline
    attributes:
      label: Karşılaştırma taban çizgisi
      description: Hangi mevcut yöntemle kıyaslanacak?
      placeholder: |
        1. Saf pinhole + sonlu fark (mevcut tracking.py davranışı)
        2. Sabit hızlı Kalman (PLANLANIYOR)
        3. NRC düzeltmeli Kalman
    validations:
      required: true

  - type: textarea
    id: metric
    attributes:
      label: Ölçüt
      description: Hangi metrik, hangi birim, hangi kabul eşiği?
      placeholder: |
        MAE ve RMSE (m/s), 10-30 m aralığında 5 m'lik bantlarda raporlanır.
        Kabul eşiği: RMSE'de %30 azalma, Wilcoxon signed-rank p < 0.001.
    validations:
      required: true

  - type: textarea
    id: data
    attributes:
      label: Veri gereksinimi
      description: Hangi veri, ne kadar, nasıl üretilecek?
      placeholder: |
        ESP32 vidalı mil ile 5 farklı sabit hızda (0.1-0.5 m/s) 60'ar saniyelik
        yer gerçeği kaydı. Eşzamanlı video: 30 fps, 1920x1080.
        Toplam ~5 saat donanım süresi.
    validations:
      required: true

  - type: dropdown
    id: hardware
    attributes:
      label: Donanım gereksinimi
      options:
        - "Yalnızca mevcut video kayıtları"
        - "Sentetik/üretilmiş veri yeterli"
        - "ESP32 vidalı mil sistemi gerekli"
        - "Yeni donanım gerekli"
    validations:
      required: true

  - type: textarea
    id: risk
    attributes:
      label: Riskler ve sınırlamalar
      description: Hangi durumda hipotez yanlış çıkar? Ölçüm hangi koşullarda geçersiz olur?
      placeholder: |
        status olayları 5000 ms aralıklı olduğu için yer gerçeği çözünürlüğü
        0.2 Hz. 30 fps video ile eşleme hatası ±250 ms olabilir; bu da 0.5 m/s
        hızda ±12.5 cm konum belirsizliği demek.
    validations:
      required: true

  - type: textarea
    id: prior
    attributes:
      label: İlgili literatür
      description: Benzer yaklaşım yayınlanmış mı? Fark ne?
      placeholder: |
        <doi veya arXiv bağlantısı> — homografi tabanlı, yer gerçeği yok.
        Bu önerinin farkı: mikrometre düzeyinde mekatronik yer gerçeği.

  - type: checkboxes
    id: checks
    attributes:
      label: Ön kontrol
      options:
        - label: Q2_MAKALE_STRATEJISI_VE_YAPAY_ZEKA_RAPORU.md içinde zaten ele alınmamış
          required: true
        - label: Mevcut `PLANLANIYOR` bileşenlerle çelişmiyor
        - label: Donanım laboratuvarı erişimi onaylandı
```

`risk` alanı zorunlu çünkü araştırma önerilerinde en sık atlanan kısım bu.
`04_test_mimarisi/02_mock_kamera_seri_port.md` bölüm 5'te hesaplanan
"5 dakikalık testte tavan hıza ulaşılamıyor" bulgusu tam olarak bu alana
yazılması gereken türden bir sınırlama.

---

## 5. `.github/PULL_REQUEST_TEMPLATE.md`

PR şablonu Markdown kalır (GitHub, PR için Issue Forms desteklemiyor).

```markdown
<!--
Başlık biçimi: Conventional Commits ile aynı olmalı.
Örnek: fix(vision): guard against unbound track_id in detection loop
Bkz. CONTRIBUTING.md → Commit biçimi
-->

## Ne değişti

<!-- 2-4 cümle. "Kod iyileştirildi" gibi içeriksiz cümle yazmayın.
     Hangi dosya/fonksiyon, hangi davranış, hangi sonuç. -->

## Neden

<!-- Hangi issue'yu çözüyor, hangi gözlem bu değişikliği gerektirdi.
     Ölçüm/kanıt varsa buraya yazın. -->

## Değişiklik tipi

- [ ] `fix` — hata düzeltme
- [ ] `feat` — yeni işlev
- [ ] `refactor` — davranış değişmedi, yapı değişti
- [ ] `perf` — performans
- [ ] `test` — test eklendi/düzeltildi
- [ ] `docs` — doküman
- [ ] `ci` / `chore` / `build` — altyapı

## Etkilenen bileşenler

- [ ] `component/vision` — dedektör, geometri
- [ ] `component/calibration` — HFOV, ölçek referansları
- [ ] `component/tracking` — hız türetimi, filtre
- [ ] `component/telemetry` — seri port, ESP32 veri akışı
- [ ] `component/firmware` — ESP32 / PlatformIO
- [ ] `component/nrc` — nöral kalibratör
- [ ] `component/ci-cd` — workflow, linter, paketleme

## Doğrulama

<!-- Çalıştırdığınız komutları ve ÇIKTILARINI yazın. "Testler geçiyor" yetmez. -->

```bash
uv run ruff check .
# çıktısı:

uv run ruff format --check .
# çıktısı:

uv run mypy
# çıktısı:

uv run pytest tests/ -q --cov=src
# çıktısı:
```

- [ ] `ruff check .` sıfır hata
- [ ] `ruff format --check .` sıfır fark
- [ ] `mypy` sıfır hata
- [ ] `pytest tests/` tüm testler geçti
- [ ] Kapsam eşiği (%85) korundu veya düşüşün gerekçesi yukarıda yazıldı
- [ ] Yeni davranış için **yeni test** eklendi
- [ ] `PLANLANIYOR` bileşenlere dokunulmadı veya dokunulduysa ablation etkisi değerlendirildi

## Akademik etki

<!-- Yalnızca vision/calibration/tracking/nrc/telemetry değiştiyse doldurun. -->

- [ ] Bu değişiklik makalede raporlanan ölçütleri **etkilemiyor**
- [ ] Etkiliyor — yeniden ölçüm gerekli:

| Ölçüt | Önce | Sonra | Değişim |
| --- | --- | --- | --- |
| MAE (m/s) | | | |
| RMSE (m/s) | | | |
| FPS | | | |

Ölçüm koşulları (video, mesafe aralığı, kamera ayarları):

<!-- Örn: traffic_test.mp4, 10-30 m, HFOV=85° (placeholder), 30 fps -->

## Kırıcı değişiklik

- [ ] Yok
- [ ] Var — aşağıda açıklandı ve `!` ile işaretlendi (`feat(scope)!: ...`)

<!-- API imzası, birim sistemi, telemetri JSON şeması veya dosya yolu değiştiyse
     burası doldurulmalı. Birim değişikliği (cm → m) özellikle dikkat gerektirir. -->

## İlgili issue

<!-- "Closes #12" biçiminde yazın ki merge edilince issue otomatik kapansın.
     Birden fazla: Closes #12, Closes #15 -->

Closes #

## İnceleyici için notlar

<!-- Neye bakılmalı, hangi kısım tartışmalı, hangi alternatifler reddedildi ve neden. -->
```

### Şablonun tasarımı

**"Doğrulama" bölümünde çıktı yapıştırma alanı.** Yalnızca kutu işaretlemek
yeterli değil; `.qwen/QWEN.md` içindeki kanıt-öncelikli kural burada da geçerli.
"Testler geçiyor" beyanı denetlenemez, `47 passed in 2.31s` çıktısı denetlenebilir.

**"Akademik etki" tablosu.** Bu bölüm endüstriyel PR şablonlarında yok.
Gerekçe: bir `geometry.py` değişikliği makaledeki MAE/RMSE sayılarını
değiştiriyorsa, makale taslağı da güncellenmeli. Bu bağlantı PR aşamasında
kurulmazsa unutulur. `Q2_MAKALE_STRATEJISI_VE_YAPAY_ZEKA_RAPORU.md`
içindeki SOTA kıyaslama tablosu bu sütunlarla eşleşir.

**"Kırıcı değişiklik" içinde birim uyarısı.** `01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md`
bölüm 1'de cm→m geçişi kararı var. Bu geçiş yapılırken PR şablonundaki bu
satır, gözden kaçırmayı engeller.

**`Closes #` yerine `Fixes #` de geçerli.** GitHub ikisini de tanır
(`closes`, `fixes`, `resolves`). Tek bir biçim önermek tutarlılık için iyi;
`Closes` seçildi çünkü `type/bug` dışındaki issue'lar için de doğal okunuyor.

---

## 6. `CODEOWNERS`

```text
# varsayılan sahip
*                                   @OWNER

# bileşen bazında
/src/vehicle_speed/geometry.py      @OWNER
/src/vehicle_speed/nrc/             @OWNER
/firmware/                          @OWNER
/.github/                           @OWNER
/docs/infrastructure/               @OWNER
```

Tek kişilik bir projede `CODEOWNERS` gereksiz görünebilir. İki faydası var:

1. Depo büyüyüp katkıcı gelince yapı hazır olur.
2. `/.github/` satırı, CI yapılandırması değişikliklerinin otomatik inceleme
   gerektirmesini sağlar — bir katkıcı workflow'u değiştiren PR açarsa, sahip
   onayı olmadan merge edilemez (branch protection'da "require review from
   code owners" açıkken).

---

## 7. Tartışma ve yönlendirme

`config.yml` içindeki Discussions bağlantısı için depoda Discussions
etkinleştirilmeli: Settings → General → Features → Discussions.

Neden issue değil: "HFOV'u nereden bulurum?" gibi sorular hata değil, ama
issue olarak açılırsa `type/bug` listesini kirletir ve kapanmadan birikir.
Discussions'ta yanıt arşivlenir ve aranabilir kalır.

Discussions etkinleştirilmeyecekse `config.yml`'deki ilk bağlantı kaldırılmalı,
yoksa ölü bir URL gösterilir.

---

## 8. Doğrulama

Şablonlar push edildikten sonra:

```bash
# şema denetimi (yerelde, push öncesi)
uv pip install check-jsonschema
check-jsonschema --check-github-workflows .github/workflows/*.yml

# Issue Forms sözdizimi için GitHub'ın kendi denetimi vardır;
# yerelde YAML geçerliliği:
python3 -c "
import yaml, pathlib
for p in pathlib.Path('.github/ISSUE_TEMPLATE').glob('*.yml'):
    yaml.safe_load(p.read_text())
    print(f'{p}: gecerli YAML')
"
```

**YAML geçerli olması yeterli değil.** GitHub Issue Forms'un kendi şeması var
(`type: textarea` + `attributes.label` zorunlu gibi). Yanlış yazılırsa form
sessizce render edilmez ve "New issue" ekranında hiç görünmez. Push sonrası
kesin doğrulama:

```bash
# web arayüzünde: github.com/OWNER/REPO/issues/new/choose
# üç şablonun da listelendiğini gör

# veya API ile
gh api repos/OWNER/REPO --jq '.has_issues'
```

Web arayüzünde görünmeyen bir form dosyası bozuktur; `gh` ile bunu denetlemenin
doğrudan bir yolu yok. Bu, elle yapılması gereken tek doğrulama adımı.
