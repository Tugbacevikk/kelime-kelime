> **Bu depo**, `Vietnamese-speech-to-text-models` deposunun 23 Ekim 2026 (commit 2f305fd) kopyasıdır. Orası olduğu gibi kalır; burada canlı akış
> için tek aşamalı Whisper medium (GPU) ve akışlı modeller (Nemotron 3.5, seda-v0.1) denenir. Ölçümler `docs/olcumler.md` içindedir.

# Ses → Metin + Konuşmacı Ayrıştırma (Türkçe, Vietnamca)

Türkçe ve Vietnamca ses kayıtlarını metne çevirir ve her cümlenin hangi konuşmacıya
ait olduğunu bulur. Dil istekte seçilir (`tr`, `vi`) ya da sesten bulunur; konuşmacı sayısı
istekte verilmezse sesten bulunur. Yalnızca CPU'da, internetsiz sunucuda çalışacak şekilde
tasarlandı. Hedef: **1 dakikalık ses en geç 1 dakikada biter.**

Mimari, fine-tuning projesindekiyle aynı fikirde (API → kuyruk → işçi → PostgreSQL → outbox →
imzalı webhook); oradaki hatalar düzeltildi, kararlar [docs/adr](docs/adr) altında.

## Hızlı başlangıç (Windows, tek bilgisayar)

```bat
baslat.bat        :: modelleri indirir (ilk sefer), sunucuyu açar
```

Tarayıcıda **http://127.0.0.1:8001**: dosya yükle, dili seç (Otomatik / Türkçe / Vietnamca),
sonucu dinle, TXT/SRT/JSON/RTTM indir. API belgesi: http://127.0.0.1:8001/docs. (Port `.env` dosyasından değiştirilebilir).
Durdurmak: `durdur.bat`. Yerel modda Redis, PostgreSQL ya da Docker gerekmez (SQLite + iş
parçacıkları). Çok makineli tam mod: `baslat_tam.bat` ya da `docker compose up`.

## Hat

```
ses ─► 16 kHz mono ─► VAD (Silero) ─► dil kararı ─► ASR (dile göre) ─────────► kelimeler + zaman
                         │                                                            │
                         └─► konuşmacı ayrıştırma (CAM++ imzası + spektral kümeleme) ─┤
                                                                                      ▼
                                              birleştirme ─► [0.1 - 3.7] speaker_00: ...
```

| Dil | Varsayılan model | Seçenekler | Not |
|---|---|---|---|
| Vietnamca | Zipformer-68M int8 (Apache-2.0) | Zipformer-30M (CC BY-NC-ND) | Sesin tamamı en sessiz anlardan bölünür |
| Türkçe | **faster-whisper medium** (fine-tuning projesinin model ailesi, MIT) | faster-whisper small, Omnilingual-300M, Whisper-turbo | Yalnızca VAD parçaları (Whisper sessizlikte uyduruyor) |

**Ölçümler** (ayrıntı: [docs/olcumler.md](docs/olcumler.md)):
- Vietnamca, VietMed 50 dosya: WER %10,1, CER %7,3. Dil tespiti 85/85 doğru.
- Konuşmacı ayırma, 9 gerçek görüşme: DER %10,2 (sayı istekte verilirse %5,2).
- Hız: 3 dk'lık diyalog 2 çekirdekte 28 sn (oran 0,15). 1 dk ses ≈ 9 sn.
- Türkçe faster-whisper medium, **yalnızca CPU** (int8, beam 5, 6 çekirdek, Tuğba'nın i5-11400H bilgisayarı):
  ısınmış modelle 9 ölçümde oran 0,46-0,66 (hedef <= 1). Aynı bilgisayarda GPU ile 0,09-0,10.
  Ölçüm 3 dosya (~2 dk'lık), tek iş; sunucu donanımında ve kısa mesajlarda yeniden ölçülmeli
  ([docs/olcumler.md](docs/olcumler.md)). Türkçe doğruluk (WER) henüz ölçülmedi.
  Kendi makinende: `python benchmarks/kabul_testi.py --dil tr --whisper-device cpu --tekrar 3 kayit.wav`.
  Oran 1'i aşarsa `.env` içinde önce `TR_MODEL=faster-whisper-small`, o da aşarsa
  `TR_MODEL=omnilingual-300m`.

### Dil seçimi

İstekte `language=tr|vi` varsa tespit atlanır. `auto` ise önce Whisper-tiny, emin olamazsa
Whisper-small; yalnızca desteklenen diller arasında. Desteklenen dillerin toplam olasılığı
%50'nin altındaysa iş **reddedilir** (`rejected`, yanlış modelle anlamsız metin üretilmez);
güven %80'in altındaysa sonuçta `uncertain: true`.

### Konuşmacı ayrıştırma

1,5 sn'lik pencerelerden CAM++ ses imzası → NME-SC spektral kümeleme (kişi sayısını kendisi
bulur) → çok yakın kümeler birleştirilir → konuşmanın %8'inden azını kaplayan "kişi" (gülme,
gürültü) en yakın kişiye katılır. `num_speakers` verilirse tam o sayı. Üst üste konuşma ayrılmaz.

## API

Kişi başına anahtar (`API_KEYS`), erişim kaydı, kalite takibi (`GET /quality`), iş ilerlemesi ve
uyarılar: [docs/entegrasyon.md](docs/entegrasyon.md#10-i̇zleme).

API anahtarı tanımlıysa (`API_KEY`, üretimde zorunlu) istekler `X-API-Key` başlığı ister.
Bağlanacak sistem için ayrıntılı rehber (alanlar, durumlar, bildirim başlıkları, imza doğrulama
örnekleri Python/Node/C#, tekrar kuralları): [docs/entegrasyon.md](docs/entegrasyon.md).

```bash
curl -H "X-API-Key: $K" -F file=@kayit.wav -F callback_url=https://ornek.com/hook http://localhost:8001/jobs
# -> 202 {"job_id": "...", "status": "queued", "duration": 61.2, "language": "auto"}
curl -H "X-API-Key: $K" -F file=@kayit.wav -F language=tr -F num_speakers=2 http://localhost:8001/jobs
curl -H "X-API-Key: $K" -F file=@kayit.wav -F external_id=mesaj-4711 http://localhost:8001/jobs
curl -H "X-API-Key: $K" http://localhost:8001/jobs/<job_id>        # durum + sonuç
```

| Uç | Ne yapar |
|---|---|
| `POST /jobs` | ses yükle; `language` (tr, vi, auto), `num_speakers` (1–20), `callback_url`, `external_id`, `profile` (full, feedback), `notify` |
| `GET /jobs/{id}` | durum; bittiyse konuşmacılı satırlar |
| `GET /jobs?limit=20` | son işler |
| `GET /jobs/{id}/transcript` | temiz transkript: `[dd:ss - dd:ss] [Sunucu - Erkek]: metin` (`?etiket=genel\|studyo\|yok`, `bicim=txt\|json`) |
| `GET /jobs/{id}/audio` | yüklenen ses (saklama süresi dolmadıysa) |
| `DELETE /jobs/{id}` | bitmiş işi, sonucunu ve sesini siler |
| `PATCH /jobs/{id}/utterances/{n}` | n. satırın metnini ve/veya konuşmacısını düzelt (`{"text", "speaker"}`) |
| `GET /corrections` | düzeltilen satırlar (modelin ilk hali ile birlikte) |
| `GET /health` | veritabanı ve kuyruk (anahtar istemez) |
| `GET /health/ready` | hazır mı: modeller yüklü (yerel) / her kuyrukta çalışan işçi (tam) |
| `GET /metrics/prometheus` | Prometheus: iş sayıları, kuyruk derinliği ve en eski mesaj yaşı, aşama süreleri, gerçek zaman oranı, webhook, DLQ |
| `POST /webhooks/test` | `{"url"}` adresine imzalı deneme bildirimi (entegrasyon kurarken) |
| `GET /webhooks/dead`, `POST /webhooks/{id}/retry` | teslim edilemeyen bildirimler ve yeniden deneme |
| `DELETE /jobs/{id}/utterances/{n}` | uydurma satırı sonuçtan kaldır (kayıtta kalır) |
| `POST /jobs/{id}/reanalyze` | aynı sesi yeni konuşmacı sayısıyla yeniden analiz (`{"num_speakers"}`, 0 = sesten bul) |
| `POST /jobs/{id}/dataset` | kontrol edilmiş sonucu veri setinin onay kuyruğuna koy (`ENABLE_DATASET_EXPORT=1`, onay zorunlu) |
| `GET /finetune/durum` | ince ayar hazırlığı: dil başına onaylı düzeltme satırı ve süresi, eşik (500 satır, 1 saat) |
| `GET /metrics` | gecikme oranı (hedef p95 ≤ 1), kuyruk doluluğu |
| `GET /` | web arayüzü |

- **`external_id`:** entegre olan sistemin kendi kimliği. Aynısı tekrar gelirse yeni iş açılmaz,
  var olan iş `200` ile döner; ağ hatasında güvenle tekrar gönderilebilir.
- **Durumlar:** `queued` → `running` → `done` | `failed` | `rejected`.
- **Profil:** `profile=feedback` kısa ve çok sayıda gelen geri bildirim mesajları için: 10 sn'den
  kısa seste konuşmacı ayırma atlanır, Whisper beam 1. Varsayılan `PIPELINE_PROFILE=full`.
- **Bildirim olayları:** `notify=running,done,failed,rejected` (virgülle, iş başına; varsayılan
  `NOTIFY_EVENTS=done,failed,rejected`). Teslim `WEBHOOK_MAX_ATTEMPTS` (8) kez denenir, sonra "ölü".
- **Aşırı yük:** `MAX_PENDING_JOBS` (varsayılan 10000) iş beklerken yeni istek `503` + `Retry-After`
  alır. İstemci başına dakikalık sınır `RATE_LIMIT_PER_MINUTE` (varsayılan kapalı) → `429`.
- **Düzeltmeler:** arayüzde her satırda "Düzelt" var. Modelin ilk yazdığı saklanır; düzeltilen
  işin sesi 24 saatte silinmez (kayıt silinene kadar kalır), çünkü ince ayar verisidir:
  `python scripts/duzeltmeleri_disa_aktar.py --cikti duzeltmeler --dil vi` →
  `finetune/veri_hazirla.py --train duzeltmeler/liste.tsv`. Yalnızca veri setine aktarma onayı
  verilmiş işler dışa aktarılır.
- **Veri seti** (varsayılan kapalı): arayüzde "Veri setine ekle" → `DATASET_DIR/bekleyen/` →
  `python scripts/veri_seti.py listele | onayla <iş> | reddet <iş>`. Test bölümü doğrudan
  `benchmarks/diyalog_testi.py` ile, eğitim bölümü `finetune/veri_hazirla.py` ile kullanılır;
  aynı ses iki bölüme giremez.
- **Güven:** satırlarda `confidence` ve güveni 0,5 altındaki kelimelerin sırası
  (`low_confidence_words`; arayüzde altı çizili). `quality.kapsama_yuzde`: konuşmanın ne kadarı
  metne döndü; boşluklar Whisper ile yeniden denenir (güvenlik ağı).
- **Webhook** (`X-Event: job.done | job.failed | job.rejected`, `X-Delivery-Id`): en az bir kez
  gider; `WEBHOOK_SECRET` varsa `X-Signature: sha256=HMAC(secret, "<X-Timestamp>.<gövde>")`.
  Callback adresi `WEBHOOK_ALLOWED_HOSTS` ile sınırlanabilir (üretimde iç ağ adresleri kapalı).

```json
{"status": "done", "job_id": "...", "duration": 61.2, "model": "zipformer-68m",
 "language": {"code": "vi", "confidence": 0.99, "source": "tespit", "uncertain": false},
 "speakers": 2, "text": "...",
 "utterances": [{"index": 0, "speaker": "speaker_00", "start": 0.1, "end": 3.5, "text": "...",
                 "confidence": 0.93, "low_confidence_words": [2]}],
 "quality": {"kapsama_yuzde": 98.1, "kurtarilan_kelime": 0}}
```

## Canlı altyazı (mikrofon)

`http://127.0.0.1:8001/canli` ya da ana sayfadaki "Canlı akış" kutusunda **Giriş: 🎙 Mikrofon**: tarayıcı mikrofonu açar, metin
konuşurken ekrana akar. Dil seçimi (Vietnamca / Türkçe / otomatik), sunum modu (tam ekran, büyük
yazı), canlı konuşmacı etiketi, "Dosyayla test", metni indirme. Durdur'a basınca kayıt normal
analize gider (konuşmacı ayrımlı transkript). [Live-stream](https://github.com/Tugbacevikk/Live-stream)
reposundan taşındı; aynı süreç ve aynı modeller, ayrıca kurulum yok. Vietnamca'da ses parçalara
**bitişik** bölünür (hiçbir ses atılmaz): aynı kayıtlarda WER %17,1 -> %10,8
([docs/olcumler.md](docs/olcumler.md)). Protokol: [docs/entegrasyon.md](docs/entegrasyon.md#11-canlı-altyazı-mikrofon-wslive).

**Başka cihazdan** (telefon, tablet, diğer bilgisayar; aynı ağda): `.env`'de

```
HOST=0.0.0.0
HTTPS=1
```

Sunucu açılışta adresleri yazar (`adres: https://192.168.1.20:8001`); o cihazda bu adresi açın.
İlk açılışta "bağlantı gizli değil" uyarısı çıkar (sertifika bu bilgisayarın kendisinin):
"Gelişmiş > yine de devam et". Tarayıcılar mikrofonu yalnızca `https` sayfada açtığı için bu gerekli.
Windows ilk açılışta Python için güvenlik duvarı izni sorar: "Özel ağlar"a izin verin.

**Bağlantı yolu** (/canli > Ayarlar > Bağlantı): *Otomatik* — bu bilgisayarda WebSocket, başka
cihazdan WebRTC (ses Opus ile UDP'den: Wi-Fi'da paket kaybına dayanıklı); WebRTC kurulamazsa
kendiliğinden WebSocket'e döner. İkisinde de aynı bölücü ve model çalışır, doğruluk aynı.

## Canlı akış (RTP / yayın)

Arayüzde "Canlı akış" kutusu ya da `POST /streams {"kaynak": "rtp://0.0.0.0:5004", "dil": "vi"}`:
metin konuşma sırasında WebSocket ile gelir, akış bitince kayıt normal analize gider.
Telefon sesi (G.711 PCMU/PCMA) doğrudan, Opus/G.722 SDP dosyasıyla, OGG/MP3 yayınları URL ile.
Ayrıntı ve ölçümler: [docs/entegrasyon.md](docs/entegrasyon.md#9-canlı-akış-rtp-yayın),
[docs/olcumler.md](docs/olcumler.md).

```bash
python -m viet_asr.interface.akis --kaynak rtp://0.0.0.0:5004 --dil vi     # komut satırı
python scripts/rtp_gonder.py kayit.wav --hedef 127.0.0.1:5004               # deneme göndericisi
```

## Tam mod (çok süreç / çok makine)

```
istemci ─► api ─► PostgreSQL (iş) + Redis Streams: asr:lid ─► asr:vi | asr:tr ─► işçiler
                      ▲                                                            │
                      └──── sonuç + bildirim satırı AYNI işlemde (outbox) ◄─────────┘
                                        dispatcher ─► imzalı webhook (artan aralıkla tekrar)
```

- Dil başına kuyruk: yalnızca yoğun dilin işçisi artırılır (`stream_worker --kuyruk vi --surec N`).
- Çöken işçinin işi 2 dk sonra başkasına geçer; 3 denemeden sonra `asr:dlq` ve `job.failed`.
- Süreç sayısı otomatik: çekirdek / `ONNX_THREADS` (Türkçe kuyrukta `WHISPER_THREADS`, varsayılan fiziksel çekirdek) ile belleğin izin verdiğinin küçüğü
  (Docker `cpus:`/bellek sınırı dahil); seçilen sayı ve gerekçesi açılışta yazılır.
- `bakim` süreci: 24 saat sonra ses, 30 gün sonra kayıt silinir; askıda kalan iş `failed` olur.

## Veritabanı (code-first)

Şema `src/viet_asr/infrastructure/db/models.py` içinde (SQLAlchemy 2.0); değişiklikler Alembic
göçüyle, uygulama açılışta kendisi uygular. SQLite (yerel) ve PostgreSQL (tam) aynı kod.
Eski sürümün veritabanı ilk açılışta veri kaybı olmadan aktarılır.

```bash
# models.py'yi değiştir -> göç üret -> gözden geçir -> test et
set DATABASE_URL=sqlite:///data/gelistirme.db
alembic upgrade head
alembic revision --autogenerate -m "aciklama"
pytest tests/db          # göç unutulursa test_sema_modellerle_ayni kırılır
```

### PostgreSQL'e geçiş (yerel mod)

Yerel mod (`baslat.bat`) SQLite yerine PostgreSQL de kullanabilir; Redis gerekmez. Geçmiş işler,
satırlar, arayüzdeki düzeltmeler, veri seti onayları ve bekleyen bildirimler taşınır.

1. PostgreSQL'i kurun (internetsiz makineye kurulum dosyası önceden indirilir) ya da Docker ile:
   `docker run -d --name asr-pg -e POSTGRES_USER=asr -e POSTGRES_PASSWORD=asr -e POSTGRES_DB=asr -p 5432:5432 postgres:16-alpine`
   (kurulum dosyasıyla kurduysanız pgAdmin ya da `psql`'de: `CREATE DATABASE asr;`)
2. `.env`: `LOCAL_DATABASE_URL=postgresql://asr:asr@localhost:5432/asr`
3. `postgrese_gec.bat` (sunucuyu durdurur, önce sayar, sonra taşır; tekrar çalıştırmak çoğaltmaz)
4. `baslat.bat`. Eski `data/uploads/jobs.db` yedek olarak kalır.

Tam moda (`baslat_tam.bat`) geçerken aynı veritabanı `DATABASE_URL` olarak verilir.

### Yedek

Sunucu açıkken bakım günde bir veritabanı yedeği alır (`data/yedek`, 14 günden eskiler silinir,
en yeni 3 yedek her zaman kalır). Bilgisayar gece kapalıysa yedek açılınca alınır. Her yedek
alındıktan sonra okunarak doğrulanır. PostgreSQL Docker'daysa `pg_dump` konteynerin içinden
çalışır (Windows'a ayrıca kurulum gerekmez; konteyner adı `BACKUP_DOCKER_CONTAINER`, varsayılan
`asr-pg`). Kurulum dosyasıyla kurulduysa `C:\Program Files\PostgreSQL\<sürüm>\bin` kendiliğinden bulunur.

```
yedekle.bat                         # şimdi yedek al
yedekle.bat listele
durdur.bat
yedekle.bat geri-yukle data\yedek\yedek_2026-10-19_030000.dump   # önce o anki hâlin yedeğini alır
```

Yedek yalnızca veritabanıdır (işler, satırlar, düzeltmeler); ses dosyaları içinde değildir. Disk
arızasına karşı `data\yedek` klasörünü ara ara başka bir diske kopyalayın: aynı diskteki yedek
diskle birlikte gider.

## Vietnamca ince ayar (Faz 2)

Zipformer-68M tıbbi veriyle WSL2 + GPU'da ince ayarlanır; çıkan model Kapı 2'yi geçerse
`VI_MODEL=models/<klasör>` ile kullanılır. Adımlar: [finetune/README.md](finetune/README.md).
Veri eşiğini izlemek ve eğitim verisini hazırlamak için: `python scripts/ince_ayar_dongusu.py durum`
(varsayılan eşik 500 onaylı satır ve 1 saat, sezgiseldir) ve `hazirla`. Eğitim ve model kararı bu betikte
değil: eğitim `finetune/egit.py`, karar gerçek ölçüm olan `finetune/kapi2.py`.

## Rust'a geçiş

Parça parça: Python sürümü başvuru kaynağı, Rust sürümü `native/` içinde, rastgele binlerce
girdide birebir aynı sonuç şartı. İlk parça birleştirme (`pip install ./native`; kurulu değilse
Python sürümü çalışır). Sıra ve kurallar: [docs/adr/0005-rust-gecisi.md](docs/adr/0005-rust-gecisi.md).

## Geliştirme

```bash
pip install -e ".[dev,bench,server]"
python scripts/download_models.py      # models/ + MANIFEST.json (sha256; internetsiz sunucuya taşınır)
pytest                                 # birim + (modeller varsa) entegrasyon
TEST_DATABASE_URL=postgresql://postgres@localhost/asr_test TEST_REDIS_URL=redis://localhost:6379/15 pytest tests/db
lint-imports                           # katman kuralları
python benchmarks/kabul_testi.py a.wav b.wav --es-zamanli 4   # 1 dk ses <= 1 dk mı (bu donanımda)
```

Diğer ölçümler: `benchmarks/diyalog_testi.py` (DER, cpWER), `yerel_dogruluk_testi.py` (WER/CER),
`dil_tespiti_testi.py`, `konusmaci_testi.py`, `turkce_dogruluk_testi.py`.
Kendi Türkçe kayıtlarınızla doğruluk: TSV'ye `dosya` ve `metin` (doğru metin) yazın,
`python benchmarks/yerel_dogruluk_testi.py kayitlar.tsv --dil tr` (üretimdeki hattan geçer).
HTTP API ve kuyruğa karşı yük (4000 mesaj projeksiyonu): `python benchmarks/yuk_testi.py --api http://127.0.0.1:8001
--ses-dosyasi kayit.wav --ornek 30 [--anahtar ...]`; gerçek konuşma şarttır (sentetik ton hızı abartır), ortalama
mesaj süresi gerçek mesajlarınıza yakın olmalı, 30'dan az örnekte projeksiyon güvenilir sayılmaz.
VietMed'den bölüm bazında test seti: `scripts/vietmed_hazirla.py`.

```
src/viet_asr/
  domain/          veri tipleri; dış kütüphane yok
  application/     iş kuralları (TranscribeAndDiarize, LanguageResolver, FusionEngine, SubmitJob...)
                   ve portlar; sherpa-onnx, SQLAlchemy, Redis, FastAPI bilmez (lint-imports)
  infrastructure/  sherpa-onnx ve faster-whisper adaptörleri, kümeleme, db/ (modeller + göçler +
                   depo), Redis Streams, iş parçacığı kuyruğu, webhook, Rust birleştirme sarmalayıcısı
  interface/       api.py, local_server.py, stream_worker.py, dispatcher.py, bakim.py, cli.py,
                   static/index.html; composition.py hepsini aynı şekilde kurar
native/            Rust çekirdeği (PyO3)
finetune/          Vietnamca ince ayar hattı ve Kapı 2
benchmarks/ scripts/ tests/{unit,db,integration} docs/{adr,olcumler.md,entegrasyon.md}
```

CI (GitHub Actions): Linux (PostgreSQL + Redis ile), Windows (yerel mod), Rust (eşlik testi).

## Yol haritası

- [x] Faz 0 — hat, dil seçimi, konuşmacı sayısı otomatik, DER/cpWER ölçütleri, VietMed bölümlü test seti
- [x] Faz 1 — API, Redis Streams, PostgreSQL/SQLite (code-first), outbox + imzalı webhook, güvenlik,
      saklama, `/metrics`, web arayüzü, kabul testi, CI, ADR'ler
- [x] Türkçe — faster-whisper medium varsayılan, small seçenek (kelime zamanlı)
- [x] Faz 2 hazırlığı — ince ayar hattı + Kapı 2 (eğitim GPU'lu makinede çalıştırılacak)
- [x] Türkçe medium hızı (oran 0,94) ve 4000 mesaj çift işleme testi — fine-tuning projesinde ölçüldü (olcumler.md)
- [ ] Türkçe gerçek kayıtlarla WER/DER ölçümü
- [ ] İnce ayarın çalıştırılması ve Kapı 2 kararı
- [x] Rust adım 2a: `EnergyChunker` (enerji kesimi)
- [x] Rust adım 2b: `tokens_to_words` (gruplama Rust'ta, yazım düzeltmesi Python'da)
