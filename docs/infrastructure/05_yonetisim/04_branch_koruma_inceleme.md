# Branch Koruma ve Kod İnceleme Süreci

**Araç durumu (doğrulandı):** `gh 2.102.0` kurulu. Uzak depo yok, dolayısıyla
koruma kuralları henüz uygulanamaz. Bu rapor kural setini ve uygulama
yöntemini tanımlar.

Branch protection ayarları **YAML ile ifade edilemez**. Depo web arayüzünden
veya `gh api` ile yapılır. Bu, altyapının "kod olarak" tutulamayan tek parçası
— bu yüzden ayarlar hem betikle uygulanabilir hem de bu belgede kayıtlı
olmalı.

---

## 1. Branch modeli

```mermaid
gitGraph
    commit id: "chore: init repo"
    commit id: "feat: package structure"
    branch develop
    checkout develop
    commit id: "feat(telemetry): pyserial reader"
    branch fix/unbound-track-id
    checkout fix/unbound-track-id
    commit id: "fix(vision): guard track_id"
    checkout develop
    merge fix/unbound-track-id id: "squash merge"
    checkout main
    merge develop id: "v0.1.0" tag: "v0.1.0"
    checkout develop
    branch exp/nrc-prototype
    commit id: "feat(nrc): 5-input residual net"
    commit id: "test(nrc): tensor contract"
    checkout develop
    merge exp/nrc-prototype id: "squash merge"
```

| Branch | Amaç | Koruma |
| --- | --- | --- |
| `main` | yayınlanabilir durum, tag'ler buradan | en sıkı |
| `develop` | entegrasyon, günlük iş | orta |
| `feat/*`, `fix/*`, `refactor/*`, `ci/*`, `docs/*` | tek iş | yok |
| `exp/*` | araştırma dalları; merge edilmeyebilir | yok |

`exp/*` dalı bu depoya özgü. NRC mimarisi denemeleri veya ablation varyasyonları
`develop`'i kirletmeden yürütülmeli. Bir `exp/*` dalı hiç merge edilmeyebilir —
bu başarısızlık değil, araştırmanın normal sonucu. Negatif sonuç da
`docs/paper/` altında belgelenmeli.

### İki branch gerçekten gerekli mi?

Tek kişilik projede `main` + `develop` ikilisi çoğu zaman fazlalık. Ama bu
projede somut bir gerekçe var: **makale sürümü**. Makale gönderildiğinde
`main` üzerindeki tag (`paper/sensors-v1`) dondurulmalı, ama geliştirme devam
etmeli. `develop` bu ayrımı sağlar.

Alternatif: tek `main` + tag'ler. Daha basit, ama "yayınlanabilir durum" ile
"günlük iş" ayrımı kaybolur. Karar: iki branch, `03_ci_cd/04_release_versiyonlama.md`
içindeki `paper/*` tag mekanizmasıyla birlikte.

---

## 2. Koruma kuralları

### 2.1 `main` için

```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["All checks"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
```

Alan alan gerekçe:

| Alan | Değer | Neden |
| --- | --- | --- |
| `strict` | `true` | PR merge'den önce `main` ile güncel olmalı; aksi halde "yeşil görünüp merge sonrası kırılan" durum oluşur |
| `contexts` | `["All checks"]` | `01_pipeline_mimarisi.md` bölüm 10'daki `summary` job'ı. Matris hücrelerini tek tek zorunlu yapmak kırılgan |
| `enforce_admins` | `true` | Depo sahibi de kurallara tabi. Kapalıysa tek kişilik projede koruma fiilen yok demektir |
| `required_approving_review_count` | `1` | Kendi PR'ını onaylayamazsın; bu, "dışarıdan bir göz" zorunluluğu getirir |
| `dismiss_stale_reviews` | `true` | Onaydan sonra yeni commit gelirse onay düşer |
| `require_code_owner_reviews` | `true` | `CODEOWNERS` içindeki `/.github/` satırı, CI değişikliklerini korur |
| `required_linear_history` | `true` | Merge commit yok, yalnızca squash veya rebase. Geçmiş okunabilir kalır |
| `allow_force_pushes` | `false` | `main` tarihi yeniden yazılamaz |
| `allow_deletions` | `false` | `main` silinemez |
| `required_conversation_resolution` | `true` | Çözülmemiş inceleme yorumu varken merge engellenir |

**`enforce_admins: true` tek kişilik projede kendini kilitler.** Kendi PR'ını
onaylayamayan tek kişi, kendi PR'ını merge edemez. Üç çıkış yolu:

1. Kuralı kapatmak → koruma anlamsızlaşır
2. İkinci bir hesap/ortak → akademik projede danışman veya lab arkadaşı
   reviewer olarak eklenebilir
3. `bypass_pull_request_allowances` ile belirli kişilere istisna

**Öneri: seçenek 3**, danışman (Ersan Hoca) reviewer olarak eklenir:

```json
"bypass_pull_request_allowances": {
  "users": [],
  "teams": [],
  "apps": ["dependabot"]
}
```

Dependabot'a istisna verilmesi pratik: bağımlılık güncellemeleri için insan
onayı beklemek birikme üretir. Ama `required_status_checks` Dependabot için de
geçerli kalır — CI yeşil olmadan merge olmaz.

### 2.2 `develop` için

```json
{
  "required_status_checks": { "strict": false, "contexts": ["All checks"] },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": false
}
```

`develop`'de onay zorunlu değil, ama CI zorunlu. Gerekçe: günlük iş akışını
kilitlemeden "kırmızı kod `develop`'e girmez" garantisi verir.

`strict: false` çünkü `develop` sürekli hareket halinde; her PR'ın güncel
olmasını zorunlu kılmak gereksiz rebase yükü üretir.

### 2.3 Uygulama betiği

`scripts/apply_branch_protection.sh`:

```bash
#!/usr/bin/env bash
# Branch koruma kurallarını uygular.
# KULLANIM: ./scripts/apply_branch_protection.sh OWNER/REPO [--dry-run]
#
# UYARI: bu betik uzak depo ayarlarını DEĞİŞTİRİR ve depo sahibini de
# kurallara tabi tutar (enforce_admins). --dry-run ile önce önizleyin.
set -euo pipefail

REPO="${1:?Kullanım: $0 OWNER/REPO [--dry-run]}"
DRY_RUN="${2:-}"

main_rules() {
cat <<'JSON'
{
  "required_status_checks": {"strict": true, "contexts": ["All checks"]},
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  },
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true,
  "bypass_pull_request_allowances": {"users": [], "teams": [], "apps": ["dependabot"]}
}
JSON
}

develop_rules() {
cat <<'JSON'
{
  "required_status_checks": {"strict": false, "contexts": ["All checks"]},
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": false
}
JSON
}

for branch in main develop; do
    echo "=== $branch ==="
    payload="$("${branch}_rules")"
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "$payload"
    else
        gh api --method PUT "repos/${REPO}/branches/${branch}/protection" \
            --input - <<< "$payload"
        echo "$branch koruması uygulandı."
    fi
done

echo
echo "Doğrulama:"
for branch in main develop; do
    gh api "repos/${REPO}/branches/${branch}/protection" \
        --jq '{branch: "'"$branch"'", checks: .required_status_checks.contexts, admins: .enforce_admins.enabled}'
done
```

Bu betik **uzak depo ayarlarını değiştirir** ve geri alma yolu
`--method DELETE` ile kuralı silmektir. `--dry-run` olmadan ilk kez
çalıştırılmamalı.

### 2.4 Web arayüzü yolu

Betik çalışmazsa (API sürüm farkı, izin eksikliği):

```text
Settings → Branches → Add branch ruleset (veya Add rule)
  Branch name pattern: main
  ☑ Require a pull request before merging
      Required approvals: 1
      ☑ Dismiss stale pull request approvals when new commits are pushed
      ☑ Require review from Code Owners
  ☑ Require status checks to pass before merging
      ☑ Require branches to be up to date before merging
      Status checks: "All checks" (arama ile ekle — workflow en az bir kez
                     çalışmış olmalı, yoksa listede görünmez)
  ☑ Require conversation resolution before merging
  ☑ Require linear history
  ☑ Do not allow force pushes
  ☑ Do not allow deletions
  ☑ Include administrators
```

**Status check adının listede görünmesi için workflow'un en az bir kez
çalışmış olması gerekir.** GitHub, check adlarını geçmiş koşulardan toplar.
Bu yüzden sıra: workflow push → bir PR aç → koşu tamamlansın → sonra koruma
kuralı ekle. Ters sırada yapılırsa "All checks" seçilemez.

---

## 3. Kod inceleme kontrol listesi

`.github/PULL_REQUEST_TEMPLATE.md` içindeki doğrulama bölümü PR sahibinin
sorumluluğu. Aşağıdaki liste **inceleyicinin** sorumluluğu ve
`CONTRIBUTING.md`'ye eklenmeli.

### 3.1 Genel

- [ ] Değişiklik PR açıklamasında anlatılanla örtüşüyor mu? Açıklamada yazan
      ama kodda olmayan (veya tersi) kısım var mı?
- [ ] Diff'in boyutu incelemeye uygun mu? 500+ satırlık PR bölünmeli.
- [ ] Commit mesajları Conventional Commits biçiminde mi?
- [ ] İlgisiz dosya değişikliği karışmış mı? (özellikle `.pio/`, `*.pt`,
      `uv.lock` beklenmedik yerlerde)

### 3.2 Doğrulama kanıtı

- [ ] "Doğrulama" bölümünde **komut çıktısı** var mı, yalnızca kutu işareti mi?
- [ ] Çıktıdaki test sayısı, değişen kodla tutarlı mı? (2 satır değişip
      "142 passed" yazıyorsa şüphelen)
- [ ] Yeni davranış için yeni test var mı? Test, hatanın **önceki** halini
      yakalıyor mu? (test'i eski kodla çalıştırınca düşmeli)

### 3.3 Bu projeye özgü

- [ ] Birim değişikliği var mı? `cm`/`m`, `cm/s`/`m/s` karışması
      `01_dizin_yapisi/02_modul_sinirlari_veri_kontratlari.md` bölüm 1'e aykırı.
- [ ] `cv2.imshow` `visualization.py` dışında mı?
- [ ] `torch`/`ultralytics` modül seviyesinde mi import edilmiş?
      (fonksiyon gövdesinde olmalı)
- [ ] Yapılandırma değeri koda gömülmüş mü? (özellikle fiziksel sabitler)
- [ ] `PLANLANIYOR` bileşenlere (Kalman, NRC) dokunulmuşsa, ablation etkisi
      değerlendirilmiş mi?
- [ ] `vision`/`calibration`/`tracking`/`nrc`/`telemetry` değiştiyse
      "Akademik etki" tablosu doldurulmuş mu?

### 3.4 Firmware

- [ ] `firmware/include/telemetry_protocol.h` ile
      `src/vehicle_speed/telemetry/protocol.py` sabitleri eşleşiyor mu?
- [ ] `Serial.print` ile elle JSON kurulmuş mu? (ArduinoJson tercih edilir)
- [ ] `loop()` içinde bloklayan çağrı var mı? Adım kaybı riski
      (`02_kod_kalitesi/04_firmware_c_statik_analiz.md` bölüm 4.2)
- [ ] Yorumlar kodun gerçek davranışını mı anlatıyor?
      (`main.cpp`'deki "Programı sonlandır" yorumu yanlış — bölüm 4.3)

### 3.5 Güvenlik

- [ ] `torch.load(..., weights_only=True)` kullanılmış mı?
- [ ] Sabit kodlanmış yol/kimlik bilgisi var mı?
- [ ] CLI'dan gelen dosya yolu denetlenmiş mi?
- [ ] `subprocess` çağrısı `shell=True` ile mi?

---

## 4. İnceleme süreci

### 4.1 PR açan taraf

1. Branch'i `develop`'den aç (`main`'den değil,除非 release hazırlığı).
2. Değişikliği küçük tut: bir PR = bir mantıksal iş.
3. `pre-commit run --all-files` yerelde temiz geçsin.
4. `gh pr create --fill` ile aç, şablonu **tam** doldur.
5. CI kırmızıysa, inceleme istemeden önce düzelt. Kırmızı PR'a inceleme
   istemek inceleyicinin zamanını boşa harcar.

### 4.2 İnceleyen taraf

| Yorum öneki | Anlamı | Merge engeller mi |
| --- | --- | --- |
| `nit:` | tercih, zorunlu değil | hayır |
| `suggestion:` | somut alternatif | hayır |
| `question:` | açıklama isteniyor | hayır |
| `blocker:` | düzeltilmeden merge edilmez | evet |
| `security:` | güvenlik kaygısı | evet |

Önek sözleşmesi, "bu yorumu uygulamak zorunda mıyım?" belirsizliğini kaldırır.
Öneksiz yorumlar varsayılan olarak `nit:` sayılır.

### 4.3 Merge

```bash
gh pr merge --squash --delete-branch
```

Squash tercih edilmesinin gerekçesi: `required_linear_history: true` ile
`main`'de tek satırlık, anlamlı commit'ler kalır. Squash commit mesajı PR
başlığından gelir — bu yüzden PR başlığı Conventional Commits biçiminde
olmalı (`02_issue_pr_sablonlari.md` bölüm 5).

`--delete-branch` merge edilen branch'i siler. `exp/*` dalları için
**kullanılmamalı**: negatif sonuçlanmış bir deney dalı, ileride "bunu
denemiştik" diye bakmak için değerli.

---

## 5. Zorunlu kontroller ve `cancelled` durumu

`01_pipeline_mimarisi.md` bölüm 7'de `cancel-in-progress: true` tanımlandı
(`main` hariç). İptal edilen koşu `cancelled` sonucu verir ve `summary` job'ı
bunu başarısızlık sayar:

```bash
if echo "$results" | grep -qE '"result": *"(failure|cancelled)"'; then
```

Senaryo: PR'a iki commit arka arkaya push edildi. İlk koşu iptal edilir, ikinci
koşu tamamlanır. `required_status_checks.strict: true` olduğu için GitHub en
**güncel** commit'in koşuluna bakar — iptal edilen eski koşu merge'i bloklamaz.

Ama `strict: false` olsaydı (develop), iptal edilmiş bir koşu "pending" gibi
davranıp merge'i kilitleyebilirdi. `develop`'de `strict: false` seçildiği için
bu risk var; pratikte `develop`'e merge eden kişi aynı kişi olduğu için
kilitlenme elle aşılabilir (`enforce_admins: false`).

---

## 6. Doğrulama

```bash
# kuralların gerçekten uygulandığını gör
gh api repos/OWNER/REPO/branches/main/protection \
  --jq '{checks: .required_status_checks.contexts, strict: .required_status_checks.strict, admins: .enforce_admins.enabled}'

# force push gerçekten engelleniyor mu (test branch'inde dene, main'de DEĞİL)
git checkout -b test/force-push-check
git commit --allow-empty -m "test"
git push -u origin test/force-push-check
git commit --amend --no-edit
git push --force-with-lease origin test/force-push-check   # bu branch korumalı değilse başarılı olur

# korumalı branch'te dene (bir PR açıp):
# beklenti: "protected branch hook declined"

# check adlarının görünürlüğü
gh run list --limit 5
gh api repos/OWNER/REPO/commits/main/check-runs --jq '.check_runs[].name'
```

Son komut, `required_status_checks.contexts` içine yazılacak adların gerçekte
ne olduğunu gösterir. Beklenen: `All checks`. Workflow'daki `summary` job'ının
`name:` alanı değiştirilirse koruma kuralı da güncellenmeli — ikisi ayrı
yerlerde tutulduğu için sessizce ayrışabilir. Bu yüzden `name: All checks`
satırına dokunulmamalı ve bu belgeye not düşüldü.

---

## 7. Açık noktalar

1. **Tek kişilik projede 1 onay zorunluluğu.** `enforce_admins: true` +
   `required_approving_review_count: 1` kombinasyonu, ikinci bir reviewer
   olmadan merge'i imkânsız kılar. Danışman veya bir lab arkadaşı reviewer
   olarak eklenmeli. Eklenemezse `main` için onay sayısı 0'a düşürülüp yalnızca
   CI zorunlu tutulmalı — bu, "koruma yok"tan iyidir.
2. **`rulesets` vs `branch protection`.** GitHub'ın yeni `rulesets` API'si daha
   esnek (koşullu kurallar, bypass listeleri). Bu rapor eski
   `branch protection` API'sini kullanıyor çünkü `gh api` ile doğrudan
   uygulanabilir ve daha yaygın belgelenmiş. `rulesets`'e geçiş ileride
   değerlendirilebilir.
3. **İmzalı commit zorunluluğu.** Yapıştırılan plan "Signed Commit şartı"
   diyordu. `required_signatures: true` ayarı var, ama tek kişilik akademik
   projede maliyeti faydasını aşıyor (GPG anahtarı yönetimi, her makinede
   kurulum). Karar: **zorunlu tutulmaz**, `07_uygulama_yol_haritasi/02_kabul_kriterleri.md`
   içinde bilinçli olarak kapsam dışı bırakıldığı kaydedildi.
