# Ölçümler

Her sayı bir betikten gelir; betik adı ve veri yanında yazılıdır. Donanım: 2 çekirdekli
sanal makine (Xeon), ONNX_THREADS=2, yalnızca CPU.

## VietMed örneği — 50 dosya, 77,2 dk (12 Ekim 2026)

Kullanıcının `vietmed_1_2dk` klasöründen rastgele 50 dosya (seed 42), 8 kHz, 1–2 dk'lık
birleştirilmiş kayıtlar. Roller: Doctor 33, Host 6, Narrator 6, Podcaster 3, Lecturer 2
(hasta kaydı çıkmadı). **Uyarı:** 626 dosyalık küme VietMed train/dev/test bölümlerini
karıştırıyor; Zipformer-30M VietMed ile eğitildiği için onun sonucu iyimser olabilir.

### ASR — `benchmarks/yerel_dogruluk_testi.py`

| Parçalama | Model | WER (hece) | CER | Eksik hece | RTF |
|---|---|---|---|---|---|
| VAD (Silero, eşik 0,5) — eski | 68M | %20,8 | %18,2 | %13,1 | 0,025 |
| VAD eşik 0,2 | 68M | %15,6 | – | %7,3 | 0,031 |
| VAD yok, sabit 15 sn | 68M | %10,8 | – | %1,9 | 0,025 |
| **Tüm ses, en sessiz anda kesim (15 sn)** | **68M** | **%10,1** | **%7,3** | %1,8 | 0,026 |
| Tüm ses, en sessiz anda kesim (15 sn) | 30M | %9,7 | %7,6 | %1,6 | 0,024 |

Bulgu: VAD müzik altındaki / gürültülü konuşmayı atıyordu (en kötü dosyada 95 sn'nin
yalnızca 3,8 sn'si ASR'a gitti). Zipformer sessizlik ve müzikte metin uydurmuyor
(eklenen hece +%0,5), bu yüzden ona artık sesin tamamı veriliyor.
30M ile 68M arasındaki 0,4 puan fark, 30M'in VietMed'i eğitimde görmüş olmasıyla
açıklanabilir; seçim tarafsız VIVOS testine bırakıldı.

Bellek: parçaları hepsini birden çözmek 92 sn'lik seste 1,7 GB'a, 77 dk'lık testte 6 GB'a
(süreç öldürüldü) çıkıyordu. 10 sn'lik gruplarla 460 MB ve daha hızlı (RTF 0,029; 60 sn
gruplarda 2,2 GB, RTF 0,043).

### Dil tespiti — `benchmarks/dil_tespiti_testi.py`

50/50 doğru (vi), 1 dosya Whisper-small'a devredildi, dosya başına 0,29 sn.

### Konuşmacı ayrıştırma — tek kişilik dosyalarda fazladan kişi

Her dosya tek konuşmacının birleştirilmiş kayıtları; doğru cevap çoğunlukla 1.

| Küçük küme kuralı | Gerçek: 1 kişi bulunan | Sentetik 28 diyalog: doğru sayı |
|---|---|---|
| en az 1 sn | 34/50 | 27/28 |
| en az 1,5 sn ve konuşmanın %8'i | **48/50** | **28/28** |

## VietMed hasta örneği — 35 dosya, 54,1 dk (13 Ekim 2026)

15 hasta, 13 doktor, 7 sunucu kaydı; VietMed_011 görüşmesinde sunucu, doktor ve hasta aynı kayıtta.
Hastaların çoğu telefonla bağlanıyor (8 kHz, hat gürültüsü).

### ASR

| Model | Genel WER | Doktor | Sunucu | Hasta | Hasta (duraksama sesleri çıkarılınca) |
|---|---|---|---|---|---|
| 68M | %10,5 | %6,8 | %7,0 | %17,3 | %14,7 |
| 30M | %10,0 | %7,3 | %6,7 | %15,5 | – |

Hasta hatalarının bir kısmı modelin "ờ", "ơ", "à" gibi duraksama seslerini yazmasından:
referansta hastaların bunları yazılmamış (en sık eklenen hece "ờ", 100 kez). Bunlar
çıkarılınca da hasta %14,7: kendiliğinden konuşma ve yöresel ağız gerçekten daha zor.

### Dil tespiti

35/35 doğru, 1 dosya Whisper-small'a devredildi.

### Konuşmacı ayrıştırma — gerçek seslerden kurulan diyaloglar

`konusmaci_testi.py --kaynak tsv`: 4 gerçek konuşmacının (sunucu, doktor, iki hasta)
8 sn'lik kesitlerinden 1-3 kişilik diyaloglar, duraksamalı ve duraksamasız, 2 tohum = 56 diyalog.

| İmza modeli | Eşik | Sayı doğru | Cümle doğru kişide | Sayı verilirse cümle | RTF |
|---|---|---|---|---|---|
| ERes2Net base (önceki) | 0,65 | 16/56 | %80,8 | %86,2 | 0,049 |
| ERes2Net base | 0,75 | 40/56 | %85,6 | %86,2 | 0,049 |
| ERes2NetV2 | 0,70 | 40/56 | %87,1 | %87,2 | 0,103 |
| TitaNet large | 0,80 | 38/56 | %85,0 | %84,5 | 0,055 |
| **CAM++ common** | **0,70** | **47/56** | **%91,7** | %89,9 | **0,030** |

CAM++ 0,70 ile tek kişilik dosyalarda: hasta 10/15, diğer roller 68/70 tek kişi buldu.
Sentetik (temiz TTS) seste 22/28 (eşik 0,65'te 25/28): ayar gerçek sese göre yapıldı.
**Sınır:** veri küçük (4 konuşmacı); eşik 0,65/0,70/0,75 arasında sonuç çok değişiyor,
daha fazla gerçek diyalogla yeniden ayarlanmalı. Hasta seslerinde tek kişiyi iki kişi
sanma hâlâ var (5/15).

## Sentetik diyaloglar — `benchmarks/konusmaci_testi.py`

Piper sesleri (vi ve tr, her dilde 3 ses), 1–3 kişi, 12 konuşma sırası.

| Yöntem | Sayı doğru | Cümle doğru kişide | RTF |
|---|---|---|---|
| pyannote seg-3.0 + WeSpeaker (eski) | 8/14 | %81,0 | 0,121 |
| parça imzası + kümeleme (yeni) | 14/14 | %100 | 0,037 |
| aynısı, sıra değişiminde duraksama yok | 14/14 | %100 | 0,046 |

## Türkçe kayıtlar — 22 dosya, 34 dk (13 Ekim 2026)

Kullanıcının gönderdiği set: 16 podcast/diyalog parçası (1,7–2,2 dk, 16 kHz), 1 müşteri
hizmetleri diyaloğu (1 dk), 2 telefon kaydı (8 kHz, 17 sn ve 63 sn), 3 adet 1 sn'lik test sesi
(saf ton). **Doğru metin yok**: doğruluk yalnızca modellerin birbirine göre farkıyla ve gözle
değerlendirildi. 2 çekirdekli bulut makinesi; dil verilmedi (sesten bulundu).

| | Omnilingual-300M | Whisper-turbo | faster-whisper small (proje, önceki sistem) |
|---|---|---|---|
| Hız, ~2 dk'lık dosya (oran) | 0,26–0,29 | 0,58–0,83 | 0,5–0,7 (kendi bilgisayarında, önceki hatla) |
| Hız, 17 sn telefon kaydı | 0,96 | 1,14 | 0,70 |
| Dil tespiti | 19/19 tr | 4/4 tr | — |
| 1 sn test sesi | düzeltmeden önce "ừ" uydurdu; düzeltmeden sonra boş | boş | "Altyazı M.K." uydurdu |

- **Kısa ses:** 17 sn'lik telefon kaydında sürenin 5 sn'si dil tespiti. Whisper-tiny 8 kHz
  telefon sesinde Türkçeye %36 verdi (desteklenen dillerin toplamı %50'nin altında), ikinci
  kademe Whisper-small %98 ile buldu. Dil istekte verilirse bu süre sıfır.
- **Uydurma:** VAD hiç konuşma bulmazsa artık ASR çalışmıyor (önceden Zipformer/Omnilingual'e
  sesin tamamı veriliyordu).
- **Kaçırılan konuşma:** önceki hat (faster-whisper small) bir podcast parçasında 6–30. sn
  arasını tamamen atladı (181 kelime; Omnilingual 230, turbo 236).
- **Modeller arası fark** (turbo'ya göre WER; turbo doğru kabul edilmez, yalnızca yön gösterir):
  podcast Omnilingual %20 / small %26; müşteri hizmetleri Omnilingual %29 / small %18;
  telefon kayıtlarında üçü de birbirinden %59–78 farklı → 8 kHz telefon sesinde hiçbirine
  güvenilemez, doğru metinle ölçülmeli.
- **Konuşmacı:** 18 dosyada 2 kişi bulundu (önceki hat aynı dosyalarda 1–4 arası). Müşteri
  hizmetleri diyaloğunda iki ses çok benzer (merkez uzaklığı 0,345, birleştirme eşiği 0,35):
  tek kişi çıktı; `num_speakers=2` verilince cümleler doğru ayrıldı. Eşiği 0,34'e indirmek bu
  dosyayı düzeltiyor ama 17 sn'lik telefon kaydını 6 kişiye bölüyor; değiştirilmedi.

## Kelime güveni — VietMed 8 dosya, 2808 hece (14 Ekim 2026)

Zipformer parça olasılıklarından kelime güveni (exp(log olasılık ortalaması)), doğru metinle
hizalanarak:

| eşik | işaretlenen | işaretlilerde hata | diğerlerinde hata | yakalanan hata |
|---|---|---|---|---|
| < 0,3 | %0,7 | %42,9 | %11,9 | %3 |
| < 0,5 (seçilen) | %3,0 | %36,5 | %11,3 | %9 |
| < 0,7 | %17,0 | %21,8 | %10,1 | %31 |

Model aşırı emin (transducer'larda bilinen durum): güven hatayı tek başına bulmaz ama
işaretli kelime 3 kat daha sık yanlış. Arayüzde 0,5 altı altı çizili gösterilir; düzeltme
yapan kişi önce onlara bakar. Genel hata (%12,1) burada kapsama ölçüsünden bağımsızdır.

## Başka alfabeden uydurma — Türkçe 22 dosya (14 Ekim 2026)

Omnilingual (1600 dil) 3174 kelimenin 5'inde başka alfabe kullandı: podcast sonundaki müzikte
Devanagari "कजतच", Arapça tek harfler ve Kiril harf karışmış "nasыsыn" (nasılsın). Artık
(normalizer_for): hiç Latin harf içermeyen kelime atılır; karışık kelimede Latin'e benzeyen
Kiril harfler düzeltilir ("nasısın"), kelime kaybolmaz.

## Araya sıkışmış kısa konuşmacı düzeltmesi — 9 Vietnamca diyalog (14 Ekim 2026)

A, kısa B, A dizisinde B'yi A'ya vermek (fine-tuning projesindeki "sandwich" düzeltmesi):

| eşik | DER | cpWER |
|---|---|---|
| kapalı | %10,64 | %32,01 |
| 1,0 sn | %10,44 | %31,82 |
| 1,5 sn | %10,14 | %31,54 |
| 2,0 sn | %10,07 | %31,35 |

Hiçbir dosya kötüleşmedi, ama **kapalı bırakıldı**: diyaloglar kesitlerden kurulduğu için
gerçek kısa araya girmeler ("ừ", "evet") yok; düzeltme onları siler ve bu sette zararı
görünmez. Kesintisiz gerçek görüşme kaydıyla ölçülünce `FUSION_SANDWICH_MAX` ile açılabilir.

## Henüz yapılmadı

- VIVOS test (tarafsız) — `benchmarks/dogruluk_testi.py`; bu ortamdan Hugging Face'e erişim yok.
- Türkçe doğru metinli ölçüm (FLEURS tr ya da elle yazılmış metin) — `benchmarks/turkce_dogruluk_testi.py`, `yerel_dogruluk_testi.py --dil tr`.
- Gerçek, kesintisiz çekilmiş doktor-hasta görüşmesi (şimdiki diyaloglar kesitlerden kuruldu;
  üst üste konuşma yok).

## Kişi sayısı tahmini: silhouette (fine-tuning projesi) ve NME-SC — 14 Ekim 2026

Aynı CAM++ gömmeleri (önbellek), yalnızca kişi sayısını bulan adım değişiyor. Doğru tahmin sayısı:

| Set | Silhouette (proje) | NME-SC (burada) |
|---|---|---|
| Tek kişilik kayıt | 34/84 | 72/84 |
| Gerçek diyalog | 6/9 | 7/9 |
| Kesit | 11/56 | 34/56 |
| Sentetik diyalog | 14/28 | 26/28 |

Silhouette tek kişiyi hiç bulamaz (en az 2 küme ister); projedeki eşikle de 1'e düşmesi zor.
Ayrıca köşegeni sıfırlanmamış uzaklık matrisinde sklearn hata veriyor. Alınmadı.

"Küçük konuşmacıyı koru" (SpeakerMerger): hasta setinde 7 → 6. Alınmadı, ayar olarak kaldı.

## Türkçe hız ve 4000 mesaj — fine-tuning projesindeki ölçümler (Tuğba'nın bilgisayarı)

Kaynak: Tugbacevikk/fine-tuning README ve docs/benchmarks (gerçek Türkçe görüşmeler, 4 iş
parçacığı, GPU yok). Bu sistemde yeniden ölçülmedi; proje sahibinin onayıyla kabul edildi.

| Model | Medyan oran | En kötü oran | 1 dk ses |
|---|---|---|---|
| faster-whisper medium | 0,94 | 0,98 | ~56–58 sn |
| faster-whisper small | 0,23 | 0,28 | ~14–16 sn |

- Medium hedefin (≤ 1) içinde ama sınıra yakın. O ölçümde sıcaklık yedeği her bölümde
  çalışıyordu; burada yalnızca boşluk kurtarmada çalıştığı için oran aynı ya da daha düşük olmalı.
- 4000 mesaj: sahte modellerle 4 eşzamanlı işçi, çift işleme yok (test_worker_accuracy.py).
  Süre tablosu small modelin oranından hesaplanmış (10 sn'lik 4000 mesaj, 4 işçi ≈ 22 dk);
  medium ile aynı yük yaklaşık 4 kat sürer. `/metrics` kuyrukta bekleme süresini gösterir.

## Türkçe yavaşlık — 15 Ekim 2026 (Tuğba'nın bilgisayarı, i5-11400H 6 çekirdek/12 iş parçacığı)

2 dk Türkçe diyalog, medium, dil sesten: **296 sn (oran 2,46)**, kapsama %83. Projede aynı
model 0,94. Kodda bulunan iki neden:

1. **İş parçacığı:** Whisper ONNX modelleriyle aynı bütçeyi alıyordu (`ONNX_THREADS=2`). Bu
   ayar ani yükte 6 işi paralel çalıştırmak için; tek iş gelince 10 iş parçacığı boş kalıyordu.
   Projede 4'tü. Artık ayrı: `WHISPER_THREADS` (varsayılan fiziksel çekirdek = 6), aynı anda
   12 / 6 = 2 Whisper çözümü. Ani yükte toplam iş hızı aynı kalır (işlemci yine dolu), tek
   mesajın süresi kısalır. (19 Ekim: Windows'taki CTranslate2 çökmesi için model kilitlendi,
   çözümler artık sırayla; tek çözüm yine 6 iş parçacığıyla.)
2. **Güvenlik ağı:** kelime çıkmayan her boşluk tek tek, 4 sıcaklıkla ve her sıcaklıkta 5
   örnekle (best_of) çözülüyordu; Whisper her çağrıda 30 sn'lik pencere işlediği için 2 sn'lik
   boşluk bile 30 sn'lik ses kadar sürüyordu. Kapsama %83 olduğu için (birçok boşluk) en
   pahalı yol çalıştı. Artık: tüm boşluklar tek toplu çözüm (sıcaklık 0); hâlâ boşların en
   uzun 4'ü sıcaklık 0,2/0,4, tek örnek, beam 1.

Medium modeli bu ortamda indirilemediği için yeni süre burada ölçülemedi; sonuçta artık
aşama süreleri (`timings`) var ve arayüzde "Aşamalar" olarak görünüyor.

## Temiz transkript ve ses perdesi — 16 Ekim 2026

- **Ses perdesi (F0) doğrulaması:** Piper sesleri — Vietnamca 3 kadın sesi 233–267 Hz (kadın),
  Türkçe dfki ve fahrettin 105 Hz (erkek), fettah 192 Hz (tek tepeli dağılım, oktav hatası değil;
  ses gerçekten tiz ya da adı yanıltıcı — dinlenerek doğrulanamadı). Eşikler bu yüzden
  ≤155 erkek / ≥185 kadın, arası belirsiz (yanlış etiket yerine etiketsiz).
- VietMed 9 diyaloğun doğru konuşmacı aralıklarında F0: 97–136 Hz ve 185–233 Hz iki öbek,
  4 konuşmacı 163–180 arası (belirsiz). Sunucuların bir kısmı kadın (011_a 207 Hz): rolden
  cinsiyet çıkarmak yanlış olurdu.
- 9 diyalog (cümle sınırı ve bileşik kelime listesi eklendikten sonra): kişi 7/9, DER %8,8
  (önce %9,0), cpWER %28,6, WER %12,5 — kayıttaki metin değişmedi.
- **Sınır:** VietMed'deki TV programlarında konuşmacılar az duraklıyor; duraklamaya dayalı
  noktalama uzun cümleler üretiyor. Dil bilgisine göre noktalama ve bağlama göre kelime
  düzeltme ("con" -> "câu") dil modeli ister; kurala dökülünce başka kayıtları bozuyor.

## Konuşmacı penceresi 1,5/0,75 sn -> 1,0/0,4 sn (Tuğba, 16 Ekim) — sert montaj geçişleri

| | 1,5 / 0,75 / 0,5 (önceki) | 1,0 / 0,4 / 0,4 (yeni) |
|---|---|---|
| Sentetik, sıra değişiminde duraksama yok (14 diyalog) | kişi 12/14, cümle %96,4 | **kişi 14/14, cümle %100** |
| 9 Vietnamca diyalog | 7/9, DER %8,8, cpWER %28,6 | 7/9, DER %8,8, cpWER %28,5 |
| En kötü oran (9 diyalog, 2 iş parçacığı) | 0,24 | 0,30 |

Ani konuşmacı değişiminde açık kazanç, gerçek diyaloglarda kayıp yok; bedeli biraz daha fazla
imza hesabı. Yeni değerler kaldı.

## Birleştirme mesafesi 0,35 -> 0,28 denemesi (16 Ekim) — alınmadı

| | 0,35 | 0,28 |
|---|---|---|
| Tek kişilik kayıt (kayıtlı imzalar) | 73/85 | 70/85 |
| Sentetik diyalog (kayıtlı imzalar) | 26/28 | 22/28 |
| Sentetik, duraksamasız (yeni pencere) | 14/14, %100 | 13/14, %98,8 |
| 9 Vietnamca diyalog | 7/9, DER %8,8 | 7/9, DER %8,8 |

Hiçbir sette iyileşme yok, üçünde kötüleşme. Varsayılan 0,35 kaldı; `DIARIZER_MERGE_DISTANCE`
ayarı korundu (sesleri çok benzer kişiler için, ama önce `num_speakers` denenmeli).

## Birleştirme mesafesi 0,15 + yeni özdeğer kuralı (16 Ekim) — alınmadı

Kayıtlı imzalarla: tek kişilik 73 -> 64/85, kesit 34 -> 29/56, sentetik 26 -> 5/28; gerçek
diyalog 7 -> 8/9, DER %17,1 -> %13,5. Gerçek diyaloglarda kazanç var ama diğer üç sette büyük
kayıp. Yeni özdeğer kuralı 0,35 ile öncekiyle birebir aynı sonucu veriyor (zararsız, kaldı).
Varsayılan 0,35; benzer sesli kayıtlar için `DIARIZER_MERGE_DISTANCE` ya da `num_speakers`.

## Canlı akış (RTP) — Vietnamca, 16 Ekim 2026

2 VietMed kaydı (~93 sn), `scripts/rtp_gonder.py` ile gerçek zaman hızında G.711 PCMU (8 kHz,
20 ms paket), alıcı `interface/akis.py`; 3 akış aynı anda, 2 çekirdek. WER kayıt metnine göre.

| | 0249 (doktor) | 0249, %2 kayıp + %5 sırası bozuk | 0467 (hasta) |
|---|---|---|---|
| Yalnızca VAD parçaları (ilk sürüm) | %13,8 | %10,3 | %21,0 |
| Bitişik parçalar | %7,4 | %6,7 | %17,9 |
| **Bitişik, en az 5 sn (son)** | **%7,2** | **%6,0** | **%16,7** |
| Aynı kayıt dosyadan (8 kHz telefon sesi) | %4,5 | — | %14,0 |
| Özgün 16 kHz dosya | %6,4 | — | %13,1 |
| Kesin metin gecikmesi p95 / en çok | 1,34 / 1,80 sn | 1,33 / 2,02 sn | 0,99 / 1,17 sn |
| Ara sonuç gecikmesi p95 | 1,04 sn | 1,25 sn | 1,07 sn |

- İlk sürümün hataları çoğunlukla silinen kelimeydi (35/58): VAD müzik altındaki ve kısık
  konuşmayı atıyor; dosya hattında da bu yüzden Zipformer'a sesin tamamı veriliyordu.
- Kalan fark (2,7-3 puan) çoğunlukla eklenen dolgu sesleri ("ờ", "ừm", "à"): kayıt metninde
  yoklar. Temiz transkript bunları zaten süzüyor; ham canlı metinde duruyorlar.
- Paket kaybı ve sıra bozulması WER'i bozmadı (kayıp yeri sessizlikle doluyor, sıra tamponu
  diziyor). Arayüz denemesi: 25 sn akış, ara sonuçlar ~1 sn'de, bitince tam analiz kuyruğa girdi.

### SDP ile L16 ve Opus (aynı gün)

Aynı doktor kaydı (93 sn), SDP'den biçim ve port: L16 16 kHz bizim alıcı, Opus 32 kbit/sn ffmpeg;
iki akış aynı anda, ara sonuç açık.

| | WER | Kesin metin p95 / en çok | Ara sonuç p95 |
|---|---|---|---|
| L16 (WAV'ın ham hali) | %6,2 | 2,8 / 3,8 sn | 0,5 sn |
| Opus (OGG'nin içindeki ses) | %6,0 | 3,2 / 3,9 sn | 0,7 sn |
| (karşılaştırma) G.711 8 kHz | %7,2 | 1,3 / 1,8 sn | 1,0 sn |

- 16 kHz'lik biçimler telefon sesinden (8 kHz) ~1 puan daha doğru.
- İlk denemede iki parçanın kesin metni 6,5 ve 8,5 sn gecikti: konuşmacı 13 sn duraklamadan
  konuştu, parça 15 sn sınırında zorla kesildi ve kesim noktası 6-9 sn geride kaldı. 4 kayıtla
  çevrimdışı benzetim: sınır 15 -> 8 sn, kesim parçanın son %40'ında: WER %11,0 -> %10,8, kesin
  metin en geç 7,1 -> 3,0 sn. Ara sonuçlar bu sırada ekranda (~0,6 sn).
- Son parçanın gecikmesi akışın bittiğinin anlaşılma süresidir (`idle_timeout`, burada 2 sn).
- ffmpeg'in akış başında 5 sn biçim araması SDP'de biçim yazılı olduğu için kapatıldı
  (ilk ses 0,06 sn'de geliyor); paket kesilince ffmpeg kendiliğinden çıkmadığı için okuma
  zaman aşımı eklendi.

### Mikrofon canlı altyazısı: Live-stream bölücüsü mü, bitişik bölücü mü (16 Ekim)

Live-stream reposu taşınırken aynı 4 VietMed kaydı (aynı Zipformer-68M) iki bölücüyle canlı
hızda çözüldü:

| Bölücü | WER | Çözülen ses |
|---|---|---|
| Live-stream (VAD: 700 ms sessizlikte cümle biter, sessizlik atılır) | %17,1 | %85-98 |
| Bitişik (5-8 sn, duraklamada kes, hiçbir ses atılmaz) | **%10,8** | %100 |

- Fark, VAD'ın cümle sınırında kestiği kısık hece ve sözcüklerden geliyor (Zipformer bağlamı
  da kaybediyor). Vietnamca canlı altyazı bu yüzden bitişik bölücüyle; Türkçe (Whisper) VAD'la
  kalıyor, çünkü Whisper sessiz parçalarda cümle uyduruyor.
- Tarayıcı denemesi (Chromium, sahte mikrofon, 0249 kaydı, 100 sn): ara sonuç ve kesin satırlar
  aktı, gecikme göstergesi 1,0-1,7 sn, Durdur'dan sonra kayıt toplu işe gitti ve tamamlandı,
  konsolda hata yok.

### Mikrofon: WebSocket mi, WebRTC mi (16 Ekim)

Aynı Zipformer, "Dosyayla test" (kayıt gerçek zaman hızında, tarayıcıdan), iki iletim yolu:

| Kayıt | WebSocket (16 kHz ham ses) | WebRTC (Opus/UDP) |
|---|---|---|
| vietmed_0249 (93 sn) | WER %5,5 | %5,7 |
| vietmed_0362 (92 sn) | %12,2 | %10,0 |
| Kesin satır gecikmesi (0249, 9 satır, 2 çekirdekli deneme makinesi) | 0,7–3,6 sn | 0,8–3,4 sn (ort. ~0,2 sn fazla) |

- Opus sıkıştırması doğruluğu düşürmedi (fark, aynı sesin parçalara farklı yerden bölünmesinden).
- Başka cihaz denemesi: sunucu `HOST=0.0.0.0 HTTPS=1`, sayfa ağ adresinden (`https://192.0.2.2`)
  açıldı; "Otomatik" WebRTC'yi seçti, ana sayfadaki Mikrofon girişi ve /canli ikisi de çalıştı,
  oturum sonunda kayıt toplu işe gitti, konsolda hata yok.
- İki sunucu ve iki tarayıcı aynı anda çalışırken (2 çekirdek) gecikme 5 sn'ye çıktı: gecikme
  işlemci yüküne bağlı, iletim yoluna değil.

### Model kilidi ve canlı altyazı (19 Ekim)

Zipformer'a ve Whisper'a eklenen kilit (Windows'ta CTranslate2 0xc0000005 çökmesi için) canlı
altyazıyı, aynı modeli kullanan toplu iş bitene kadar bekletiyordu. 4,6 dk Vietnamca toplu iş
sürerken canlı oturum (Vietnamca, 60 sn, 2 çekirdek):

| | Ortalama gecikme | En kötü |
|---|---|---|
| Toplu iş yok | 1,9 sn | 3,3 sn |
| Toplu iş sürerken, kilitli | 5,4 sn | 17,5 sn |
| Toplu iş sürerken, Zipformer kilitsiz | 2,6 sn | 3,8 sn |

- Zipformer (sherpa-onnx) eşzamanlı çağrıda güvenli: 4 iş parçacığı x 40 çağrı, çökme yok,
  sonuçlar tek tek çözümle birebir aynı. Çökme CTranslate2'deydi (Whisper); Zipformer kilidi
  kaldırıldı.
- Whisper kilidi duruyor (çökme Windows'ta, burada yeniden üretilemedi; ayrıca faster-whisper'ın
  BatchedInferencePipeline'ı kelime zamanı için ortak durum tutuyor). Bedeli: Türkçe canlı altyazı
  Türkçe toplu iş sürerken bekler.

### Konuşmacı ayırma hızı ve verilen konuşmacı sayısı (19 Ekim)

2 dk Vietnamca seste (2 çekirdek) aşamalar: konuşma bulma 1,3 sn, metne çevirme 6,8 sn, konuşmacı
ayırma 7,5 sn. Konuşmacı ayırmanın neredeyse tamamı 283 pencerenin (1 sn, 0,4 sn kayma; bitişik
parçalarda sessizlik yok denecek kadar az) ses imzası. Kümeleme 0,01-0,2 sn.

| Ses imzası hesabı | Süre |
|---|---|
| Önceki: her pencere 2 iş parçacığıyla, sırayla | 7,3 sn |
| Her pencere 1 iş parçacığıyla, 2 pencere aynı anda | 4,1 sn |
| 9 gerçek diyalog (toplam): önceki -> yeni | 136 -> 77 sn (1,76 kat) |
| Aynı anda 2 iş (2 dk ses): önceki -> yeni | 13,3 -> 9,0 sn |

- İmzalar birebir aynı (benzerlik 1,000000), 9/9 diyalogda konuşmacı çıktısı aynı, DER %8,39.
  Küçük CAM++ modelinde bir pencereyi iki iş parçacığına bölmek verimsiz; aynı çekirdeklerle iki
  pencere aynı anda daha çok iş çıkarıyor (yalnızca tek iş değil, ani yükte toplam hız da artar).
- Denenip bırakılan: pencereleri tek çağrıda toplu çözmek (kaldi-native-fbank + onnxruntime).
  sherpa-onnx ile aynı ayarlara rağmen imzalar tutmadı (benzerlik 0,53-0,87); doğruluğu riske
  atmamak için alınmadı.
- Sayı verildiğinde (num_speakers) önce otomatik kümeleme denenir, o da aynı sayıyı bulduysa
  onun kümeleri kullanılır. Önbellekteki imzalarla: gerçek diyalog DER %12,3 -> %12,3, kesit
  diyalog yanlış cümle %11,7 -> %11,7, sentetik %10,9 -> %6,3. fine-tuning projesindeki "küme
  çökmesi" koruması burada gereksiz: 65 çok konuşmacılı kayıtta hiç çökme yok.


## Türkçe hız — GPU'lu geliştirme bilgisayarı (20 Ekim 2026)

Tuğba'nın bilgisayarı (i5-11400H, RTX 3050 Ti 4 GB), `viet_asr.interface.cli --lang tr`, faster-whisper
medium, CUDA FP16 (ayar verilmediği için `auto` GPU'yu seçti). Ölçümü kullanıcının yardımcısı yaptı;
JSON dosyaları kullanıcıda, bu depoda yok. Her dosya bir kez çalıştı.

| Dosya | Ses (sn) | Hat süresi (sn) | Oran | Satır |
|---|---|---|---|---|
| parca_01 | 114,3 | 22,4 | 0,196 | 14 |
| parca_02 | 101,1 | 20,4 | 0,201 | 13 |
| diyalog_parca_01 | 120,2 | 18,4 | 0,153 | 16 |
| diyalog_parca_02 | 115,8 | 16,0 | 0,138 | 7 |

- Hiçbir dosyada 10 sn'den uzun boşluk yok. Aşama süreleri (parca_01): Whisper 17,7 sn, konuşmacı
  ayırma 4,0 sn, VAD 0,4 sn.
- **Bu GPU sonucudur; GPU'suz hedefin kanıtı değil.** CPU ölçümü için `--whisper-device cpu`
  (ya da `WHISPER_DEVICE=cpu`) ile `benchmarks/kabul_testi.py` ya da CLI. 20 Ekim'e kadar
  CLI ve kabul testi `WHISPER_*` ayarlarını yok sayıyordu; düzeltildi, eski ölçümler bu yüzden `auto` idi.
- Doğruluk (WER) ölçülmedi: doğrulanmış referans metin yok.

## Türkçe hız — yalnızca CPU, Whisper paketleme düzeltmesi (20 Ekim 2026)

Tuğba'nın bilgisayarı (i5-11400H, 6 çekirdek, 16 GB RAM; ölçümde boş RAM 5-6 GB), `WHISPER_DEVICE=cpu`,
faster-whisper medium int8 beam 5, `whisper_threads=6`. Ölçümü kullanıcının yardımcısı yaptı; JSON'lar
kullanıcıda. Chrome ve Docker kapalıydı. Dosyalar: parca_01, diyalog_parca_01, diyalog_parca_02 (~2 dk).

**Neden yavaştı:** VAD 2 dk'lık sesi 19 (diyalog_parca_01) ve 39 (parca_01) küçük parçaya bölüyordu ve
her parça ayrı bir 30 sn'lik Whisper penceresinde çözülüyordu; süre parça sayısıyla orantılıydı
(39/19 = 2,05; ölçülen oran 2,86/1,39 = 2,06). Düzeltme: yakın parçalar tek pencerede paketlenir
(`FasterWhisperAsr.pack_clips`; en fazla 29,5 sn, aradaki boşluk en fazla 3 sn).

| Ölçüm (CPU medium) | Önce | Sonra |
|---|---|---|
| parca_01, CLI, soğuk (model her çalıştırmada yüklenir) | 2,86 | 1,21 / 0,80 / 0,70 |
| diyalog_parca_01, CLI | 1,39 | 0,58 |
| Isınmış model, `kabul_testi.py --tekrar 3`, 9 çalıştırma | – | 0,455-0,659 (en kötü 0,659) |

- Isınmış çalıştırmalar tekrar boyunca düştü (parca_01: 0,66 → 0,61 → 0,51); kararlı değer daha düşük olabilir.
- Soğuk ilk çalıştırma 1,21 oldu; nedeni ölçülmedi. Sunucuda modeli açılışta yükleyip ısıtın (`WARMUP_LANGS`).
- GPU medium: 0,09-0,10 (önce 0,14-0,20). Small, CPU, beam 1: 0,19-0,23.
- Medium, CPU, beam 1: iki dosyada da `0xC0000005` ile çöktü (paketlemeden önce); yeniden denenmedi, kullanmayın.
- Metin: CPU ile GPU çıktısı arasındaki kelime farkı paketlemeden sonra %0,5-6,0, önce %6,4-14,1
  (3 dosya; birbirine kıyas, doğruluk değil). Paketleme çıktıyı eski koda göre %7,7-21 değiştirdi;
  hangisinin referansa daha yakın olduğu doğrulanmış metin olmadığı için bilinmiyor.
- **Sınırlar:** tek bilgisayar, 3 dosya, tek iş. Sunucu donanımı, kısa mesajlar ve 4000 mesajlık ani yük ölçülmedi.

## Türkçe canlı akış — yalnızca CPU, iki aşamalı (omnilingual anlık + medium düzeltme) (20 Ekim 2026)

Makine: i5-11400H (6 çekirdek/12 iş parçacığı), 15,8 GB RAM, Windows 11. Ses: `parca_01.wav` (114,3 sn),
`scripts/dis_servis_ornek.py ... --hiz 1 --parca 1` (gerçek zamanlı push). Ayarlar: `STREAM_TWO_STAGE=1`,
`TR_MODEL=faster-whisper-medium`, `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`, `WHISPER_BEAM_SIZE=5`.
Ölçen: Antigravity (kendi çıktısı); ham JSON'lar kullanıcının bilgisayarında. Ben yalnızca raporu okudum,
sayıları yeniden hesaplamadım.

| Düzen | Boş RAM | Koşu | duvar_sn | Anlık toplam gecikme p50/p95/max (sn) | Düzeltme gecikmesi p50/max (sn) |
| --- | --- | --- | --- | --- | --- |
| Yalnız medium (iki aşamasız) | 2,88 GB | 1 | 374,1 | 91,4 / 104 / 109,8 | — |
| Yalnız omnilingual | — | 1 | 120,3 | 3,2 / 5,2 / 6,5 | — |
| İki aşamalı, grup bekleme 2 sn (35d709d) | 5,3–5,6 GB | 2 | 150,4 ve 143,2 | 6,1–7,0 / 9,6–18,6 / 11,6–20,4 | 27,8–58,6 / 37,8–83,4 |
| İki aşamalı, grup bekleme 10 sn + 24 sn erken çözme (54de2e9) | 6,8–7,1 GB | 3 | 175,3–184,9 | 7,2–13,2 / 24,4–30,3 / 26,2–32,3 | 77,7–82,1 / 97,1–100,1 |

Bulgular:
- Yalnız medium canlıya yetişmiyor (akış süresi/ses süresi 3,27). İki aşamalı düzen gecikmeyi 91 sn'den 6–13 sn'ye indirdi.
- DÜZELTME (aynı gün, sonraki ölçüm): 54de2e9'daki 10 sn bekleme ile 2 sn bekleme arasında fark ölçülemedi.
  Aynı kodla 2 sn bekleme yeniden ölçüldü: duvar 161–181 sn, düzeltme p50 74–88 sn (aşağıdaki tablo, A düzeni);
  yani yukarıdaki 143–150 sn / 28–58 sn değerleri yeniden üretilemedi (neden bilinmiyor: koşu değişkenliği
  olabilir). Medium canlıya yetişmediği için kuyruk hep dolu; gruplar bekleme süresinden değil birikmeden oluşuyor
  (bu bir yorum, doğrudan sınanmadı). Sabitler eski davranışa (2,0 / 29,5) çevrildi, sebep "kötüleştirdi" değildi.
- Anlık metin, medium arka planda çalışırken yavaşlıyor (omnilingual tek başına 3,2 sn; iki aşamalıda 6–13 sn).
  Neden: CPU çekişmesi. İş parçacığı sayısı denemesi yapılmadı.
- Boş RAM koşular arasında 2,9 / 5,3–5,6 / 6,8–7,1 GB. Koşullar tam eşit değil; 2 sn düzeninin yeniden ölçümü gerekir.
- Metin kalitesi: yalnızca 7 parça el ile karşılaştırıldı. WER ölçülmedi (insan doğrulamalı referans yok).

### Aynı düzen, medium iş parçacığı sayısı ve düzeltme modeli (20 Ekim 2026, 3 koşu/düzen)

Ses ve ortam aynı (`parca_01.wav`, 114,3 sn, i5-11400H, boş RAM 6,2–7,6 GB). Ölçen: Antigravity; ham özet JSON'lar
kullanıcının bilgisayarında, ben raporu okudum.

| Düzen | duvar_sn | duzeltme_pencere | Anlık toplam gecikme p50 (sn) | Anlık p95 (sn) | Düzeltme gecikmesi p50 / max (sn) |
| --- | --- | --- | --- | --- | --- |
| A medium, 6 iş parçacığı | 161,0 / 163,6 / 181,4 | 5, 5, 5 | 8,4 / 8,5 / 16,4 | 20,8–29,6 | 74–88 / 94–106 |
| B medium, `WHISPER_THREADS=2` | 174,0 / 175,1 / 188,7 | 5, 5, 3 | 2,4 / 2,6 / 8,5 | 4,0 / 4,3 / 11,5 | 84 / 84 / 147; max 105–160 |
| C medium, `WHISPER_THREADS=3` | 170,5 / 190,7 / 194,0 | 5, 5, 4 | 3,8 / 6,2 / 11,1 | 6,4 / 16,3 / 16,7 | 84–124 / 105–137 |
| D small, 6 iş parçacığı | 127,0 / 131,7 / 140,8 | 14, 15, 15 | 5,4 / 5,6 / 11,9 | 8,8 / 11,7 / 20,2 | 12,7 / 13,6 / 21,2; max 21–35 |

Bulgular:
- Medium düzeltmesi hiçbir iş parçacığı ayarında canlıya yetişmiyor; akış, ses bittikten ~45–80 sn sonra bitiyor.
- `WHISPER_THREADS=2` anlık metni omnilingual tek başına hızına (p50 ~2,5 sn) getiriyor (2 koşuda; 3. koşu 8,5 sn),
  bedeli düzeltmenin daha geç gelmesi. Koşular arası değişkenlik büyük (n=3).
- Small düzeltmesi ses bittikten ~13–27 sn sonra bitiyor (akış süresi/ses 1,11–1,23); düzeltme gecikmesi p50 13–21 sn.
  Small'ın metin kalitesi medium'a göre ölçülmedi.
- `WHISPER_THREADS` genel bir ayar; dosya (toplu iş) hızına etkisi bu ölçümde bakılmadı.

## Türkçe dosya hızı — CPU ayar taraması (21 Ekim 2026)

Makine: i5-11400H (6 fiziksel/12 mantıksal çekirdek), 15,8 GB RAM, Windows 11, yalnızca CPU. Model:
`faster-whisper-medium`, int8. Ses: `parca_01.wav` (114,3 sn) + `parca_02.wav` (101,1 sn).
Komut: `benchmarks/kabul_testi.py ... --dil tr --whisper-device cpu`. Ölçen: Antigravity; ben raporu okudum,
ham JSON'ları görmedim. Boş RAM koşulara göre 3,4–7,7 GB değişti.

**1. Tek tek ayar (parça başına 3 tekrar, toplam oran = işlem süresi / ses süresi):**

| Ayar | Toplam oran | Metin farkı (taban: beam 5, batch 8, 6 iş parçacığı) |
| --- | --- | --- |
| Taban | 0,688 | — |
| batch 4 / batch 16 | 0,797 / 0,755 | %0,00 |
| beam 2 / beam 3 | 0,577 / 0,677 | %3,0 / %0,9 |
| 4 / 12 iş parçacığı | 0,637 / 0,578 | %0,00 |
| 2 eşzamanlı iş (3'er iş parçacığı) | toplam 0,592; tek tek 1,09–1,25 | %0,00 |
| 3 eşzamanlı iş (2'şer iş parçacığı) | toplam 0,868; tek tek 2,1–3,0 | %0,00 |

**2. Karışık sıralı (ABAB düzeni, her ayar 5 tur, aynı oturum):**

| Ayar | Koşu oranları (toplam duvar / ses) | Ortalama | En kötü tek dosya |
| --- | --- | --- | --- |
| X: 6 iş parçacığı, beam 5 | 0,703 0,669 0,643 0,612 0,823 | 0,690 | 0,846 |
| Y: 12 iş parçacığı, beam 5 | 0,686 0,752 0,867 0,792 0,777 | 0,775 | 0,892 |
| Z: 12 iş parçacığı, beam 2 | 0,607 0,633 0,683 0,738 0,584 | 0,649 | 0,826 |

Bulgular:
- 12 iş parçacığı bu makinede **daha hızlı değil**: tek tek taramadaki %16 kazanç koşu gürültüsüydü (karışık
  sıralı ölçümde Y, X'ten ortalama %12 yavaş). `WHISPER_THREADS=auto` (fiziksel çekirdek) kalıyor.
- Aynı 12 iş parçacığında beam 2, beam 5'ten 5 turun 5'inde hızlıydı (Y−Z: +0,05 ile +0,19, ortalama %16).
  Beam 2'nin 6 iş parçacığıyla ölçümü yok. X'e göre Z'nin farkı (%6) koşu gürültüsünün içinde.
- Beam 2 metni %3 değiştiriyor; hangisinin daha doğru olduğu bilinmiyor (referans metin yok, WER ölçülmedi).
- Eşzamanlı iş: toplam verim biraz artıyor ama tek tek oran 1'i geçiyor; "her ses kendi süresinde" şartını bozuyor.
- Koşu gürültüsü büyük: aynı ayarda tek koşu oranı 0,61–0,82. Tüm 15 karışık koşuda tek dosya oranı < 1
  (en kötü 0,892).
- Rapordaki "Duvar Süresi" sütunu (ör. 164,7 sn), JSON'daki `toplam_duvar_sn` (151,5 sn) ile uyuşmuyor; oranlar
  JSON'a göre hesaplanmış görünüyor. Fark nedeni bilinmiyor.

## Türkçe dosya hızı — temiz koşullarda yeniden ölçüm (21 Ekim 2026)

Aynı makine, ses ve komut (`kabul_testi.py ... --dil tr --whisper-device cpu --whisper-threads 6`, batch 8, int8).
Ölçüm sırasında başka iş çalıştırılmadı, prizde ve "En iyi performans" güç modunda. ABAB düzeni, 6 tur. Ölçen:
Antigravity (ham JSON'lar kullanıcıda; ben raporu okudum).

| Ayar | Koşu oranları (toplam duvar / ses) | Ortalama | En kötü tek dosya |
| --- | --- | --- | --- |
| X: beam 5 | 0,476 0,461 0,465 0,452 0,446 0,445 | 0,457 | 0,487 |
| Y: beam 2 | 0,369 0,373 0,377 0,362 0,354 0,351 | 0,359 | 0,414 |

- **Temiz koşulda medium CPU oranı ~0,46** (önceki ABAB ölçümünde 0,69). Önceki ölçümlerdeki büyük oynama
  büyük olasılıkla ölçüm sırasında çalışan başka işlerden geliyordu. Tur arası oynama şimdi ~%7 (önceden ~%30).
  Bu oturumdan önceki CPU tablolarındaki oranlar (ve iki aşamalı canlı akış ölçümleri) arka plan yüküyle ölçülmüş
  olabilir; kötümser sayılmalı.
- Beam 2, beam 5'ten 6 turun 6'sında hızlı (ortalama %21). Metin kelime düzeyinde sabit %3,0 / %2,9 farklı.
  Hangisinin daha doğru olduğu bilinmiyor (referans metin yok).
- Ölçüm öncesi işlemci kullanımı %7–19 idi (hedef <%5 değildi); 5. ve 6. turda sayaç hata verdi. İlk turlar
  biraz yavaş, oran turdan tura hafifçe düşüyor (X: 0,476 → 0,445). Hafif bir arka plan yükü kalmış olabilir.
- Rapordaki saat hızı sabit 2688 MHz (WMI `CurrentClockSpeed`). Bu değer çoğu zaman anlık frekansı değil
  nominal hızı gösterir; "ısıl kısma yok" kanıtı olarak zayıf. Tutarlı süreler kısma olmadığını destekliyor ama kanıtlamıyor.

## Türkçe canlı akış — temiz koşulda, iki aşamalı (21 Ekim 2026)

Aynı makine (i5-11400H, prizde, "Yüksek performans" güç modu, Chrome/Docker/WSL/VLC kapalı). Ses: `parca_01.wav`
(114,3 sn), `scripts/dis_servis_ornek.py ... --hiz 1 --parca 3` (3 sn'lik push parçaları), tek akış. Ortak ayar: CPU,
int8, beam 5, `STREAM_MAX_SPEECH=10`. Düzen başına 3 tur, karışık sıra (ABCD ABCD ABCD). Ölçen: Antigravity; ham JSON'lar
kullanıcıda, ben raporu okudum. Ölçüm öncesi işlemci %8–26 idi (hedef <%5 değildi: Antigravity IDE'nin kendi yükü).

| Düzen | Akış süresi / ses | Sesten sonra bitiş (sn) | Anlık metin gecikmesi p50 / p95 / max (sn) | Düzeltme gecikmesi p50 / p95 / max (sn) | Medium/small çağrısı |
| --- | --- | --- | --- | --- | --- |
| A: yalnız Omnilingual | 1,045 | 5,1 | 1,1 / 3,6 / 3,8 | — | 0 |
| B: + medium, 6 iş parçacığı | 1,13 | 14,9 | 1,8 / 4,0 / 4,3 | 27 / 52 / 58 | 8 |
| C: + medium, 2 iş parçacığı | 1,19 | 21,2 | 1,3 / 3,7 / 3,95 | 48 / 70 / 76 | 5–6 |
| D: + small, 6 iş parçacığı | 1,065 | 7,5 | 1,1 / 3,6 / 3,8 | 5,3 / 12,0 / 12,8 | 17 |

- Temiz koşulda medium düzeltmesi tek akışta **yetişiyor**: akış sesten ~15 sn sonra bitiyor (önceki kirli
  ölçümde 45–80 sn). Önceki "medium canlıda yetişmiyor" sonucu arka plan yükünden etkilenmişti.
- `WHISPER_THREADS=2` temiz koşulda gereksiz: anlık metin zaten hızlı (p50 1,8 sn) ve düzeltme daha geç
  geliyor (p50 48 sn, bitiş 21 sn). Önceki "2 iş parçacığı öner" tavsiyesi geri alındı. Varsayılan `auto` kalıyor.
- Anlık gecikme ve koşu süreleri çok tutarlı (3 turda ±%1–3).
- Ölçülmedi: birden çok eşzamanlı akış, 8 kHz telefon sesi, düzeltme metninin doğruluğu (WER).
- Aritmetik tahmin (ölçüm değil): tek bir Whisper çözümü aynı anda çalıştığı ve medium ~0,46 oranında
  olduğu için medium düzeltmesi en fazla ~2 eşzamanlı akışa yeter; small (0,19–0,23) ~4–5'e.

## Türkçe canlı akış — eşzamanlı akış sayısı (21 Ekim 2026)

Aynı makine ve koşullar (i5-11400H, prizde, "Yüksek performans", ağır programlar kapalı; Antigravity IDE
BelowNormal önceliğe çekildi; başlangıç işlemci ortalaması %8,5). Her akış farklı dosya (`parca_01..04`: 114,3 /
101,1 / 106,2 / 104,1 sn), hepsi aynı anda başlatıldı, `--hiz 1 --parca 3`, `STREAM_MAX_SESSIONS=8`, beam 5, int8,
`WHISPER_THREADS=auto`. 12 düzen/akış kombinasyonu × 2 tur, karışık sıra. Ölçen: Antigravity; ham JSON'lar
kullanıcıda, ben raporu okudum. Aşağıdaki "bitiş" değerleri, en uzun dosyanın (parca_01) turun duvar süresine göre
hesaplanan bitişidir; raporda kısa dosyaların "bitişi" tur duvar süresinden hesaplanmış olduğu için şişkin (üst sınır).

| Düzen | Akış | Bitiş (parca_01, sesten sonra sn) | Anlık gecikme p50 (ort.) | Anlık p95 / max | Düzeltme gecikmesi p50 (ort.) | Düzeltme max (sn) | Tur işlemci |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A Omnilingual | 1 | 4,9–5,2 | 1,0 | 3,5 / 3,7 | — | — | %15 |
| A | 2 | 5,0–5,3 | 1,3 | 3,1 / 4,2 | — | — | %19 |
| A | 3 | 5,6–6,0 | 2,0 | 3,6 / 4,7 | — | — | %41 |
| A | 4 | 5,4–6,1 | 2,2 | 3,7 / 4,9 | — | — | %38 |
| B +medium | 1 | 10,9–11,3 | 1,7 | 3,6 / 4,1 | 25,5 | 56 | %71 |
| B | 2 | 54,9–56,6 | 1,9 | 3,5 / 4,3 | 60 (35–86) | 105 | %66 |
| B | 3 | 65,5 | 2,6 | 3,8 / 4,6 | 91 | 129 | %71 |
| B | 4 | 65,6–66,2 | 3,6 | 5,7 / 8,0 | 119 | 172 | %79 |
| D +small | 1 | 6,4 | 1,2 | 3,5 / 3,7 | 6,8 | 51 | %39 |
| D | 2 | 15,3–16,2 | 2,1 | 3,7 / 4,6 | 17 (7–32) | 58 | %77 |
| D | 3 | 24,2–30,4 | 2,5 | 3,8 / 4,6 | 27 (8–48) | 77 | %76 |
| D | 4 | 17,8–60,2 | 3,1 | 4,4 / 5,6 | 33 (22–52) | 76 | %82 |

- **Yalnız Omnilingual 4 eşzamanlı akışı rahat taşıyor:** anlık gecikme p95 ≤3,7 sn, max ≤4,9 sn, ortalama işlemci %38.
  4'ün üstü ölçülmedi.
- **Medium düzeltmesi yalnızca tek akışta güvenilir.** 2 akışta düzeltme gecikmesi ~35–86 sn, 3–4 akışta 90–120 sn;
  4 akışta anlık metin de bozuluyor (max 8 sn). Test sesleri ~100–114 sn; daha uzun görüşmelerde birikimin büyümesi
  beklenir (ölçülmedi).
- **Small düzeltmesi 2–3 akışta kullanılabilir ama kararsız:** aynı kombinasyonda iki tur arasında düzeltme p50'si
  7 → 22 sn, 8 → 48 sn gibi büyük sapma var. Small'ın metin doğruluğu ölçülmedi.
- Bu tablo yalnızca ses uzunluğu 100–114 sn olan akışları kapsar; tek ses türü (16 kHz, podcast), 8 kHz telefon sesi değil.

## Türkçe 8 kHz telefon sesi — `akis` komutu, tek model medium (21 Ekim 2026, kullanıcının sohbetinden)

Kullanıcı kendi bilgisayarında `diyalog_parca_01.wav` dosyasını `ffmpeg -re ... -ar 8000 -ac 1 -c:a pcm_mulaw -f rtp rtp://127.0.0.1:5004`
ile gönderdi; `python -m viet_asr.interface.akis --kaynak rtp://0.0.0.0:5004 --dil tr` dinledi. Ayar: CPU, int8, `WHISPER_THREADS=2`,
tek model faster-whisper-medium (`STREAM_TWO_STAGE` bu komutta kullanılmaz). Ölçüm yapan: kullanıcı (ekran çıktısı); ben okudum.

- 1. koşu: 63,6 sn, 22 parça, gecikme p95 18,5 sn. 2. koşu: 120,2 sn, 44 parça, gecikme p95 15,9 sn.
- Gecikme yüksek: tek model medium + 2 iş parçacığı (iki aşamasız, temiz koşul ölçümündeki gibi, bu düzende medium canlıya yetişmiyor).
- Metinde hatalar var ("yüz yüze" → "100", "hocalarla" → "bacalarla", anlamsız satırlar); referans metin olmadığı için WER ölçülmedi.
- Aynı 16 kHz sesle karşılaştırma yapılmadı; 8 kHz'in tek başına etkisi ayrılamadı.

## Türkçe doğruluk — tek diyalog dosyası (22 Ekim 2026)

**Sınır:** tek kayıt (`diyalog_parca_01.wav`, 120 sn, iki konuşmacı, 187 kelime, SHA256 9A63247F…5FCC), tek konuşmacı çifti.
Referans metin elle yazılmıştır (`benchmarks/yerel_dogruluk_testi.py` normalleştirmesiyle karşılaştırıldı). Sonuçlar yön
gösterir, kesin sıralama değildir: 1 puan ≈ 2 kelime. CPU (int8), 6 iş parçacığı, sunucu kapalı, ABAB sıralı, 2 tekrar;
her ayarda iki tekrarın metni birebir aynı çıktı (deterministik). Betik Whisper'a yalnızca Silero VAD parçalarını verir
(iş hattındaki güvenlik ağı ve konuşmacı birleştirme yok).

| Model | Örnekleme | Beam | WER | CER | RTF |
|---|---|---|---|---|---|
| faster-whisper-medium | 16 kHz | 5 | %23,5 | %11,4 | 0,41 |
| faster-whisper-medium | 8 kHz | 5 | %22,5 | %13,0 | 0,35–0,39 |
| faster-whisper-medium | 16 kHz | 2 | %23,5 | %12,0 | 0,32–0,33 |
| faster-whisper-medium | 8 kHz | 2 | %20,3 | %8,8 | 0,27–0,28 |
| faster-whisper-small | 16 kHz | 5 | %25,7 | %7,9 | 0,14–0,15 |
| faster-whisper-small | 8 kHz | 5 | %28,9 | %10,5 | 0,14 |
| omnilingual-300m | 16 kHz | – | %61,5 | %32,7 | 0,11 |
| omnilingual-300m | 8 kHz | – | %40,6 | %13,4 | 0,11 |

- Medium 8 kHz telefon sesinde WER'i korudu (%20–22); small 16 → 8 kHz'de %25,7 → %28,9.
- Beam 2, bu dosyada beam 5'ten kötü değil (16 kHz'de eşit) ve %20–30 hızlı. Tek dosya olduğu için varsayılan beam 5 bırakıldı.
- 16 kHz'de small ile medium arası fark 4–5 kelime: ayırt edilemez.
- Omni yalnızca anlık önizleme işi görür; 16 kHz'de 8 kHz'den kötü çıkması bu kayıtta tekrarlandı (nedeni bilinmiyor).
- Tekrarlanan gerçek hata: "yüz yüze" → "100 100" / "yüz yüzyıl" / "yüz yüzeyitimi" (medium'da da).

> **Veri kaynağı notu:** ilk raporda sunulan hipotez metinleri (ve bunlardan hesaplanan %15–17'lik medium değerleri), aynı raporun
> kendi WER sütunuyla tutmuyordu; yeni metinlerle hesaplanan WER ilk raporun sütunuyla birebir tuttu. O ilk metinlerin kaynağı belirlenemedi;
> o değerler **geçersizdir** ve bu tabloyla değiştirildi.

## Türkçe canlı akış ve dosya hattı — tek diyalog dosyası (22 Ekim 2026, ikinci tur)

Aynı kayıt ve referans (187 kelime). **Tek dosya, yön göstericidir.** Ölçüm öncesi CPU %10–16 (geliştirme ortamı
süreçleri yüzünden %5'in altına inmedi), her koşu öncesi kuyruk boştu. Canlı akış sırasında CPU ortalaması yaklaşık %69.

**Canlı RTP (ffmpeg, G.711 PCMU 8 kHz, gerçek zaman), iki aşamalı mod açık, 4 tekrar:**

| | Sonuç |
|---|---|
| Anlık metin (omni) WER / CER | %41,2 / %14,5 (4 tekrarda aynı metin) |
| Anlık metin gecikmesi (toplam, kuyruk dahil) p50 / p95 | 1,8–1,9 sn / 3,8–4,7 sn |
| Konuşma → ekran (parça bitişine göre) p50 / p95 | 2,3–2,4 sn / 5,1–5,3 sn |
| Parça sayısı / ortalama uzunluk | 19 / 5,86 sn |
| Medium düzeltmesi ses bitişinden sonra tamamlanır | 13,6–13,9 sn |
| Düzeltmeli canlı metin WER | %26,7–27,3 |
| Akış bitince çalışan tam analiz (`son_is`) WER / CER | %20,9 / %8,7 |
| Tam analiz süresi (120 sn kayıt) | 48–51 sn |

- Parçalar ortalama 5,9 sn olduğundan, bir parçanın ilk kelimesi söylendikten yaklaşık 8 sn sonra görünür (ölçülmedi, parça uzunluğundan çıkarım).
- İki aşamalı modda parça boyutunu `STREAM_MAX_SPEECH` değil, `StreamTranscriber` içindeki `min_chunk=5 sn` ve
  `max_chunk=8 sn` sınırları belirler (kodda doğrulandı; hızlı motor omni "bitişik" mod kullanır). `STREAM_MAX_SPEECH=6/4` parça sayısını
  değiştirmedi (19). `MAX_UTTERANCE_SEC` yalnızca /canli mikrofon sayfasını etkiler.

**Dosya yükleme hattı (`POST /jobs`, dil=tr, varsayılan profil, beam 5):**

| Giriş | WER | CER | Süre |
|---|---|---|---|
| 16 kHz orijinal (2 tekrar) | %23,5 | %11,4 | 52–53 sn |
| 8 kHz kopya (2 tekrar) | %20,9 | %11,4 | 49–51 sn |

- Aynı ses için benchmark betiği (medium, beam 5, 16 kHz) iş hattıyla **birebir aynı metni ve %23,5 WER'i** verdi; iş parçacığı
  sayısı (2 ya da 6) metni değiştirmedi, yalnızca hızı: RTF 0,50 (2 iş parçacığı) / 0,41 (6). İş hattında bir hata yok.
- 8 kHz'de WER'in düşmesi (%23,5 → %20,9), güvenlik ağının 5 kelimeyi (özellikle "devam et") geri kurtarmasından geliyor
  (kurtarilan_kelime=5; 16 kHz işinde güvenlik ağı çalışmadı). Tek dosya, tesadüf olabilir.
- RTP yolu bozmuyor: aynı kaydın 8 kHz dosya olarak işi de %20,9.
- İş hattı süresi (16 kHz, 120 sn): ASR 48,7 sn, konuşmacı ayırma 2,6 sn, toplam 52,5 sn (RTF ≈ 0,44).

## Türkçe doğruluk — ikinci kayıt `diyalog_parca_04.wav` (22 Ekim 2026)

125,9 sn, 16 kHz, referans elle yazılmış, 230 kelime; ses referansın başında/sonunda fazladan konuşma içermiyor.
Aynı konuşmacı çifti olabilir (aynı diyalog serisi), bağımsız konuşmacı değil. CPU (int8), 6 iş parçacığı, ABAB, 2 tekrar
(her ayarda tekrarlar birebir aynı metin). Rakamlar Antigravity raporundan; hipotez metinleri verilen 5 koşudan 3'ü
yeniden hesaplamayla birebir tuttu, 2'si 2 kelime farkla (elle kopyalama kaynaklı olabilir), kalan modellerin metni verilmedi.

| Model | Örnekleme | Beam | WER | CER | RTF |
|---|---|---|---|---|---|
| faster-whisper-medium | 16 kHz | 5 | %15,7 | %6,3 | 0,38 |
| faster-whisper-medium | 8 kHz | 5 | %17,4 | %9,5 | 0,34–0,36 |
| faster-whisper-medium | 16 kHz | 2 | %16,1 | %6,7 | 0,28–0,30 |
| faster-whisper-medium | 8 kHz | 2 | %17,4 | %9,4 | 0,28 |
| faster-whisper-small | 16 kHz | 5 | %17,4 | %7,0 | 0,14 |
| faster-whisper-small | 8 kHz | 5 | %22,2 | %12,2 | 0,13–0,14 |
| omnilingual-300m | 16 kHz | – | %47,8 | %21,6 | 0,12 |
| omnilingual-300m | 8 kHz | – | %28,3 | %8,5 | 0,11–0,12 |

Dosya yükleme hattı (`POST /jobs`): 16 kHz %15,7, 8 kHz %17,4 (benchmark betiğiyle aynı değer), 49–51 sn (RTF ≈ 0,40).
Canlı RTP (G.711 8 kHz, iki aşamalı): anlık metin %32,6, medium düzeltmeli canlı metin %13,0, akış bitince tam analiz %17,4;
anlık gecikme p50 1,7 sn / p95 4,6 sn; düzeltme ses bitişinden ~15,6 sn sonra tamamlandı (nedeni ölçülmedi).

İki kayıtta ortak, tekrarlanan bulgular (yön göstericidir, iki kayıt):
- Medium, small'dan iyi ya da eşit; 8 kHz telefon sesinde fark açılıyor (kayıt 1: %20–22 / %29; kayıt 2: %17 / %22).
- Beam 2, beam 5'ten en fazla 1–2 kelime geride (bir koşuda önde) ve %20–30 hızlı.
- **Omni 16 kHz'de 8 kHz'den yaklaşık 19–21 puan kötü** (kayıt 1: %61,5 / %40,6; kayıt 2: %47,8 / %28,3; canlı push 16 kHz'de de %65). Nedeni
  bilinmiyor. 16 kHz kaynaklarda (mikrofon sayfası, push) anlık metin bu yüzden daha zayıf olabilir.
- Whisper "birinci sınıf" ve "elli" gibi sayıları rakamla yazıyor, normalleştirme bunu çevirmiyor; rakam farkı WER'e +1 kelime ekler.

## Omnilingual 8 kHz bantlama (22 Ekim 2026)

Canlı akışta anlık metni üreten omnilingual-300m, 16 kHz girişte 8 kHz'e göre ~20 puan kötü çıkıyordu.
`SherpaOmnilingualAsr` artık girişi 8 kHz'e indirip geri çıkarıyor (`OMNI_BANDLIMIT_HZ`, varsayılan 8000, 0 = kapalı).

Ölçüm (diyalog_parca_01, 16 kHz wav, `benchmarks/yerel_dogruluk_testi.py`, aynı betik, tek koşu; sonuçlar deterministik):

| OMNI_BANDLIMIT_HZ | WER % | CER % | RTF |
|---|---|---|---|
| 0 (kapalı) | 61,50 | 32,66 | 0,277 |
| 8000 | 40,11 | 12,80 | 0,243 |

Sınırlar: tek kayıt üzerinde doğrulandı (önceki ölçümlerde 2 kayıt ve canlı RTP aynı yönü gösterdi); farklı konuşmacılarda
ölçülmedi. Nedeni bilinmiyor (seviye normalleştirme işe yaramadı, bant sınırlama yaradı). Whisper medium düzeltmesi bundan etkilenmez.

## Vietnamca canlı akış (RTP) — rastgele 2 VietMed kaydı (23 Ekim 2026)

`random.seed(42)` ile vietmed_1_2dk'tan (626 kayıt) rol filtresi olmadan 2 dosya: vietmed_0114 (Doctor, 90,2 sn) ve
vietmed_0025 (Lecturer, 96,2 sn). Gerçek zaman hızında G.711 PCMU, tek akış, Zipformer-68M, her dosya 2 koşu
(Antigravity ölçtü; hipotez metinleri canlı koşudan, WER'i metinlerden ben yeniden hesapladım). Başlangıç CPU ~%16.

| Dosya | Canlı WER | Dosya hattı WER | Kesin metin gecikmesi p50 / p95 / en çok | Ara sonuç p95 |
|---|---|---|---|---|
| 0114 (Doctor) | %12,39 | %12,68 | 0,17 / 0,94 / 2,0 sn | 0,17 sn |
| 0025 (Lecturer) | %18,24 (benim hesabım %18,89) | %17,26 | 0,16 / 1,03 / 3,0 sn | 0,17 sn |

- İki koşu metin olarak birebir aynı (deterministik). Kuyruk bekleme 0.
- 0114'ü yeniden hesaplamam rapordaki değerle aynı; 0025'te 2 kelimelik fark var (metin aktarımı ya da normalleştirme).
- En çok gecikme ilk parçada (ilk kesim için en az 5 sn ses gerekiyor); sonrakiler ~0,1–0,5 sn.
- Hataların çoğu tıbbi terimler (beta hcg -> "dta hcv", progesterone -> "proson"), dolgu sesleri ("ờ", "à", "á") ve
  benzer sesli hece karışıklıkları. Canlı ile dosya hattı birbirine yakın: canlı akış doğruluğu bozmuyor.
- Sınır: 2 kayıt, tek akış; eşzamanlı Vietnamca akış bu turda ölçülmedi. Zipformer VietMed ile eğitildiği için sonuç iyimser olabilir.

### Canlı RTP doğrulaması — diyalog_parca_01, 2 ayar x 2 iletim x 2 tekrar (23 Ekim 2026)

Antigravity ölçtü (komut satırı `akis --dil tr`; PCMU: ffmpeg 8 kHz, L16: `rtp_gonder.py --codec l16` 16 kHz); WER'i rapordaki
tam metinlerden ben yeniden hesapladım (7 metinde rapordakiyle birebir aynı). Referans: kullanıcının insan metni (187 kelime).
Iki aşamalı: omni anlık + medium düzeltme, CPU, medium beam 5.

| İletim | OMNI_BANDLIMIT_HZ | Anlık WER | Nihai (medium) WER, 2 tekrar | Anlık gecikme p50 / p95 | Düzeltme ses bittikten sonra |
|---|---|---|---|---|---|
| PCMU 8 kHz | 0 | %41,18 | %27,27 / %24,06 | 1,9 / 4,3–4,5 sn | 14–15 sn |
| L16 16 kHz | 0 | **%65,24** | %21,39 / %22,46 | 2,2–2,3 / 5,2–5,4 sn | 16–17 sn |
| PCMU 8 kHz | 8000 | %42,25 | %24,06 / %24,06 | 1,9 / 3,6–4,1 sn | 14 sn |
| L16 16 kHz | 8000 | **%42,25** | %24,06 / %22,46 | 2,2–2,3 / 5,4–5,7 sn | 15–18 sn |

- Bantlama, 16 kHz canlı girişte anlık metni %65,2 -> %42,3 yapıyor (dosya ölçümüyle aynı yön ve büyüklük); 8 kHz girişte etkisi yok
  (%41,2 -> %42,3, bir-iki kelime). Nihai metin bantlamadan etkilenmiyor (21–27 arası, tekrarlar arası 1–6 kelime oynuyor).
- Anlık metin iki tekrarda birebir aynı; medium düzeltmesi tekrarlar arasında 1–6 kelime değişiyor (düzeltme pencereleri 9–15 adet, zamanlamaya bağlı).
- Nihai metindeki kalıcı hata: "yüz yüze" -> "100 100" / "müzik eğitimi". Medium bu ifadeyi tüm koşularda yanlış yazıyor.
- Sınır: hâlâ tek kayıt. Koşuların çoğunda akış başlamadan CPU %68–86'ydı (yalnızca ilk koşuda %15); gecikmeler bu yüzden
  temiz koşuya göre biraz kötü olabilir, WER etkilenmez. L16'da gecikme PCMU'dan ~0,3 sn yüksek (6010 paket, ek yeniden örnekleme).

### Docker'da çok akışlı canlı ölçüm, tam analiz süresi ve PostgreSQL/Redis testleri (23 Ekim 2026)

Antigravity ölçtü (Windows i5-11400H, Docker Desktop: 12 çekirdek / 7,65 GB, `docker compose`, commit 3697a74, başlangıç CPU %12);
rapordaki sayıları ben tekrar hesapladım. Yalnızca `diyalog_parca_01` (120,2 sn) ve `diyalog_parca_04` (125,85 sn) kullanıldı. Tek kişi/iki kayıt: genelleme yapmayın.

**A. Gerçek PostgreSQL 16 + Redis 7 ile `tests/db`:** 53 test, 51 geçti, 0 kalan, 2 atlandı
(`test_backup.py:69,80` — Windows ana makinede `pg_dump` yok; yedekleme testleri bu yüzden koşmadı, Docker içinde ayrıca denenmedi).

**B. Docker'da canlı akış (Türkçe, iki aşamalı, RTP/UDP):**

| Koşu | Anlık gecikme p95 / max | Düzeltme | RTP kaybı (kayıp / beklenen) |
|---|---|---|---|
| 1. akış (soğuk model) | 3,99 / 4,80 sn | 20 | 710 / 6010 = %11,8 |
| 2. akış (sıcak model) | 5,91 / 6,69 sn | 19 | 701 / 6010 = %11,7 |
| Aynı anda, akış 1 (parca_01) | 6,60 / 6,81 sn | 19 | 225 / 6010 = %3,7 |
| Aynı anda, akış 2 (parca_04, düzeltme kotası dolu, yalnız anlık) | 8,17 / 11,69 sn | yok | 1534 / 6281 = %24,4 |

- İlk akışta ilk parça 27,5 sn'de geldi (omnilingual modelinin ilk yüklenmesi); sonrakiler 8–15 sn. Yani **ilk canlı çağrıdan önce modeli ısıtmak gerekir**.
- Kota çalıştı: `STREAM_TWO_STAGE_MAX=1` olduğu için ikinci eşzamanlı akış yalnız anlık metinle çalıştı (`iki_asama_kota_doldu: true`).
- Bellek yaklaşık 2,8–3,0 GiB (api) + 1,9–2,7 GiB (worker-tr), sınır 7,65 GiB: iki akışta sığıyor, ama boşluk dar.
- **Çözülmemiş sorun — RTP paket kaybı:** Docker'da %4–24 paket kaybı var (api konteyneri 300–900 % CPU kullanırken). Önceki yerel
  (Docker'sız) ölçümlerde kayıp raporlanmamıştı. Kaybın nerede olduğu **bilinmiyor**: Docker Desktop'ın UDP aktarımı mı, gönderici mi, yoksa
  API'nin yoğunken soketi yetişememesi mi (çekirdek tamponu dolması)? Bu ölçümden ayırt edilemez. Kayıp sesi eksilttiği için Docker'daki
  kelime sayıları (143–166, referans 187) yerel ölçümle birebir kıyaslanmamalı; bu koşularda WER hesaplanmadı.
- Dikkat — koşular birbirinden temiz ayrılmamış olabilir: 1. akış bitince son kaydı normal iş olarak `worker-tr`'ye gider (~40 sn sürer).
  2. akışın ilk 10 saniyesinde `worker-tr` %538 CPU'daydı; yani 2. akışın gecikmesi (5,91 sn) önceki akışın son işiyle yarıştı.
  Aynı şekilde C bölümündeki 75 sn'lik koşunun 34,7 sn'si kuyrukta bekledi (başka bir iş işçiyi tutuyordu — doğrulanmadı).
- Rapordaki "anlık / düzeltilmiş örnek" iki ayrı akıştan (parca_04 / parca_01) alınmış; aynı konuşmanın karşılaştırması değil.
  Tablodaki "ilk parça 11,96 / 15,05 sn" değerleri gösterilen JSON'larda yok, kanıtsız.

**C. Dosya işi (tam analiz: VAD + ASR + konuşmacı ayırma), `worker-tr=1`, Docker:**

| Dosya | İşlem süresi (3 tekrar) | RTF (işlem) | Not |
|---|---|---|---|
| parca_01 (120,2 sn) | 47,0 / 41,2 / 39,1 sn | 0,33–0,39 | 2. tekrarda 34,7 sn kuyrukta bekledi → toplam 75,0 sn |
| parca_04 (125,85 sn) | 37,6 / 41,8 / 41,1 sn | 0,30–0,33 | — |

Süre dağılımı (parca_01, 1. tekrar): ASR 42,7 sn, konuşmacı ayırma 3,4 sn, VAD 0,6 sn, ses profili 0,3 sn, toplam 47,0 sn.
Şartı (işlem süresi ≤ ses süresi) rahat sağlıyor; kuyruk bekleme süresi şarta dahil edilirse tek işçiyle yük altında aşılabilir.
Sınır: yalnız 2 dakikalık kayıtlar; uzun kayıtlar ve eşzamanlı çok sayıda iş ölçülmedi.

### RTP paket kaybının kök nedeni (23 Ekim 2026)

Antigravity ölçtü (commit 26abf9f); sayaç toplamlarını ben kontrol ettim. Kayıp **Docker Desktop'tan değil, API'nin soketi okumayı bırakmasından** geliyordu:
canlı akış döngüsü (`stream.py`) hem UDP soketini okuyor hem ASR'yi aynı döngüde çalıştırıyordu; ASR sürerken soket okunmuyor, çekirdek tamponu
(WSL2'de `rmem_max` 208 KiB → soket en çok 416 KiB, ~4 sn ses) doluyor ve çekirdek paket düşürüyordu (`RcvbufErrors`).

- Windows'tan gönderimde 5800 alındı + 210 çekirdekte düştü = 6010: Docker Desktop paketi kaybetmiyor.
- Gönderici konteynerin içindeyken (Docker Desktop aktarımı yok) iki iki aşamalı koşuda 0 kayıp; ana makineden 2 koşudan birinde %3,5, ötekinde 0.
  Kayıp, işlemci başka işle (önceki akışın son işi) yarışırken ortaya çıkıyor.
- `STREAM_TWO_STAGE=0` ağır medium'u canlı döngüde doğrudan çalıştırır (yük **artar**): %48–66 kayıp. Bu üretim ayarı değildir, yalnız mekanizmayı gösterir.
- Windows göndericisi 15,6 ms zamanlayıcı yüzünden paketleri 0/31 ms patlamalarla yolluyor (ortalama 50 paket/sn doğru); kaybın nedeni bu değil.
- Raporda `1966+3940` ve `2665+3311` "6010'a eşit" denmiş; gerçekte 5906 ve 5976 (34–104 paket açıklanamıyor). Ayrıca "olay döngüsü bloklandı" ifadesi yanlış
  (okuyucu bir döngü değil, ASR ile aynı iş parçacığı).

**Düzeltme:** `RtpSource` artık soketi ayrı bir iş parçacığında sürekli boşaltıp paketleri bellekteki kuyruğa koyuyor (`tests/unit/test_rtp_reader.py`:
yavaş tüketicide 400/400 paket; eski kodda 12/400). Docker'da yeniden ölçülmedi — bu ölçüm bekliyor.

### Canlı akışta Whisper medium: CPU / GPU / tek aşama (23 Ekim 2026)

Antigravity ölçtü (commit ee66910, Docker'sız yerel sunucu, RTX 3050 Laptop 4 GB, Windows i5-11400H, RTP/PCMU 8 kHz, `diyalog_parca_01` 120,2 sn,
medium beam 5, OMNI_BANDLIMIT_HZ=8000, her ayar için ısınma + 3 tekrar). Ortalamaları ve oranı ben yeniden hesapladım. Metinlerin WER'i bir sonraki bölümde (n=1 kayıt).

| Ayar | Anlık metin gecikmesi p50 / p95 | Medium çözme p50 / max (pencere başına) | Ses bittikten sonra son metin | GPU belleği | Ortalama CPU |
|---|---|---|---|---|---|
| A: CPU, iki aşamalı | 1,85 / 3,91 sn | 6,34 / 7,59 sn | 12,42 sn | — | %58,1 |
| B: GPU, iki aşamalı | 1,04 / 2,89 sn | 0,76 / 0,88 sn | 6,88 sn | 2587 MiB | %19,3 |
| C: GPU, tek aşama (medium canlı model) | 0,53 / 0,83 sn* | 0,53 / 1,22 sn | 0,36 sn | 2171 MiB | %12,5 |

- 11 akışın hepsinde RTP paket kaybı 0: ayrı okuyucu iş parçacığı düzeltmesi yerelde işe yarıyor (Docker'da henüz ölçülmedi).
- GPU medium'u 8,3 kat hızlı çözdü (6,34 → 0,76 sn). B'de omnilingual anlık gecikmesi de düştü (1,85 → 1,04 sn p50), çünkü medium CPU'yu artık meşgul etmiyor.
- Canlı akış sürerken aynı anda 125,85 sn'lik dosya işi: GPU'da 8,1 sn (RTF 0,064), bellek hatası yok, en çok 2875 MiB. C'de dosya işi başlarken tek bir pencere 3,73 sn sürdü.
- *C'de "anlık gecikme" yalnızca parça kapandıktan sonraki çözme süresidir; kelime ekrana, konuşma 5–8 sn'lik parça dolunca gelir (ilk metin 5,9 sn'de). Tek aşamalı modda ara (interim) metin üretilmiyor. Gerçek kelime gecikmesi ölçülmedi.
- B'de "ses bittikten sonra" 6,88 sn'nin yalnızca ~0,8 sn'si medium; kalanı bekleme mantığı (REFINE_IDLE_SEC = 2 sn vb.) ve omni çözmesi. Kalemlere ayrılmadı.

### Whisper medium canlı akış: CPU / GPU, iki aşamalı / tek aşamalı (23 Ekim 2026)

Antigravity ölçtü (yerel Windows, Docker yok, RTX 3050 Laptop 4 GB, commit ee66910); ortalamaları ve WER'i ben yeniden hesapladım
(metinleri rapor dosyasından programla ayrıştırdım). Kayıt: `diyalog_parca_01` (120,2 sn, PCMU 8 kHz), referans: kullanıcının insan metni (187 kelime).
A = CPU int8, iki aşamalı; B = GPU fp16, iki aşamalı; C = GPU fp16, tek aşamalı (medium doğrudan canlı model). 3 tekrar, rtp.kayip hepsinde 0.

| Ayar | Medium pencere p50 / p95 / max | Ses bittikten sonra | Anlık WER | Nihai WER (3 tekrar) | Nihai kelime |
|---|---|---|---|---|---|
| A | 6,34 / 7,59 / 7,59 sn | 12,42 sn | %42,25 | %25,13 / %25,13 / %26,74 | 157–160 |
| B | 0,76 / 0,85 / 0,88 sn | 6,88 sn | %42,25 | %27,81 x3 | 153 |
| C | 0,53 / 0,83 / 1,22 sn | 0,36 sn | (medium) | %25,67 x3 | 179 |

- GPU belleği: B 2587, C 2171 MiB. Canlı akış sürerken parca_04 dosya işi GPU'da 8,06–8,08 sn (süre/ses 0,064), bellek hatası yok (B_ek 2875 MiB).
- A/B/C nihai WER farkı birkaç kelime (187 kelimede 1 kelime = %0,53); tek kayıtta sıralama yapılamaz.
- C'de konuşmada olmayan ifadeler çıkıyor ("hoşça kalın", "teşekkürler", "damat hadi", "bekleyin", "yüz yüzey için göndermek istiyorum evet devam etsene bizi"):
  Whisper uydurması. A/B nihai metni 30+ kelime eksik bırakıyor (silme 28–35), C 11.
- "yüz yüze" üç ayarda da "100 100" / "100" çıkıyor.
- "Ses bittikten sonra" ve gecikme değerleri parça kapandıktan sonraki bekleyiştir; kelimenin söylendiği andan ekrana gelişi değil (parça 5–8 sn ses biriktiriyor; ilk parça C'de bile 5,89 sn).
- Sınır: tek kayıt, tek konuşmacı çifti; Docker'da GPU kurulu değil, bu koşular Docker'da yapılmadı.
