"""Hattın tüm ayarları tek yerde. Kodda sabit sayı yerine buradan okunur.

Her dil bir LanguageProfile'dır: hangi ASR modeli, hangi ayarlarla.
Yeni dil eklemek = yeni bir profil eklemek; uygulama kodu değişmez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelRef:
    """Bir model paketinin dosyaları.

    kind: altyapıda hangi adaptörün kullanılacağı
          ("zipformer-transducer", "zipformer-streaming", "whisper", "omnilingual-ctc")
    files: rol -> dosya adı (klasöre göre), ör. {"encoder": "encoder.int8.onnx"}
    """

    name: str
    kind: str
    folder: Path
    files: dict[str, str]
    license: str = ""

    def path(self, role: str) -> str:
        return str(Path(self.folder) / self.files[role])

    def has(self, role: str) -> bool:
        return role in self.files and (Path(self.folder) / self.files[role]).exists()


@dataclass(frozen=True)
class WhisperParams:
    """faster-whisper (CTranslate2) ayarları. Değerler fine-tuning projesindeki Türkçe
    ayarlardan alındı; farklı olanların gerekçesi yanında."""

    device: str = "auto"              # "auto" (GPU varsa cuda, yoksa cpu), "cuda", "cpu"
    compute_type: str = "auto"        # "auto" (cuda'da float16, cpu'da int8), "float16", "int8"
    device_index: int = 0
    beam_size: int = 5
    batch_size: int = 8               # BatchedInferencePipeline: aynı anda çözülen parça
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    # Projede None idi. None iken "sessizlik olasılığı > 0,6" tek başına parçayı atar;
    # -1,0 (kütüphane varsayılanı) ile model metinden eminse parça korunur -> sessiz
    # konuşan hastanın cümlesi atılmaz.
    log_prob_threshold: float | None = -1.0
    # Projede "Türkçe ses kaydı." idi. Whisper istemi sessiz yerlerde aynen yazabiliyor;
    # dil zaten verildiği için varsayılan kapalı (istenirse açılır, çıktıdan da süzülür).
    initial_prompt: str | None = None
    hotwords: str | None = None       # boşlukla ayrılmış terimler (ör. ilaç adları)
    # aynı kelime art arda bu sayıdan fazla gelirse fazlası atılır (Whisper döngüsü)
    max_repeat: int = 2
    # Aynı modeli aynı anda kullanabilecek iş sayısı. 1 iken yerel moddaki iş parçacıkları
    # Türkçe işlerde birbirini BEKLER (CTranslate2 bir çağrıyı bir işçide çalıştırır);
    # yerel modda eşzamanlı iş sayısına eşitlenir (composition.default_config workers=).
    num_workers: int = 1
    # Bir çözümün kullandığı iş parçacığı. None: AsrParams.num_threads (ONNX modelleriyle aynı, 2).
    # Whisper medium 2 iş parçacığıyla tek işte oran ~2,5'e çıkıyordu (Tuğba'nın bilgisayarı,
    # 2 dk ses 296 sn); projede 4'tü. Ayar: WHISPER_THREADS (varsayılan fiziksel çekirdek).
    cpu_threads: int | None = None


@dataclass(frozen=True)
class AsrParams:
    num_threads: int = 2
    decoding_method: str = "modified_beam_search"  # yalnızca zipformer
    hotwords_file: str | None = None  # yalnızca zipformer; terimler BÜYÜK HARFLE
    hotwords_score: float = 1.5
    # beam genişliği (modified_beam_search). Hotword çok parçalı terimlerde ancak geniş
    # beam ile etkili: "XƠ VỮA" (V+Ữ+A) 4 yol/skor 3'te 0/5, 10 yol/skor 5'te 5/5 (13 Ekim).
    max_active_paths: int = 4
    max_word_duration: float = 1.0  # tek kelime/hece en fazla bu kadar sürebilir (sn)
    # Parçalar en fazla bu kadar toplam sesten oluşan gruplar halinde birlikte çözülür.
    # Hepsini birden çözmek belleği patlatıyor: 92 sn'lik seste 27 parça -> 1,7 GB,
    # 77 dk'lık testte 6 GB'ta süreç öldürüldü. Ölçüm (12 Ekim, Zipformer-68M):
    #   10 sn: 460 MB, RTF 0,029 | 30 sn: 1,2 GB, RTF 0,032 | 60 sn: 2,2 GB, RTF 0,043
    batch_seconds: float = 10.0
    # Whisper sessizlik/gürültüde bu cümleleri uyduruyor; parça yalnızca bundan ibaretse atılır
    drop_phrases: tuple[str, ...] = ()
    whisper: WhisperParams = field(default_factory=WhisperParams)  # yalnızca faster-whisper


@dataclass(frozen=True)
class LanguageProfile:
    code: str  # ISO 639-1: "vi", "tr"
    asr_model: ModelRef
    asr: AsrParams = field(default_factory=AsrParams)
    # "full": sesin tamamı en sessiz anlardan bölünerek ASR'a verilir (Zipformer, Omnilingual).
    # "vad": yalnızca VAD'ın konuşma dediği parçalar (Whisper: sessizlikte metin uyduruyor).
    segmentation: str = "full"


@dataclass(frozen=True)
class ChunkConfig:
    """'full' bölme: parça en fazla max_len sn, kesim en sessiz ana.
    VietMed 50 dosya, Zipformer-68M: 10 sn %10,2 | 15 sn %10,1 | 20 sn %10,0 WER."""

    max_len: float = 15.0
    min_len: float = 5.0


@dataclass(frozen=True)
class LidConfig:
    """Dil tespiti (Whisper encoder + tek adım decoder), ucuzdan pahalıya kademeli.

    stages: (encoder, decoder) çiftleri. İlki karar veremezse (ret ya da düşük
    güven) sıradaki denenir. Varsayılan: tiny (~0,25 sn) -> small (~1 sn).
    """

    stages: tuple[tuple[str, str], ...] = (
        ("models/sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx",
         "models/sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx"),
        ("models/sherpa-onnx-whisper-small/small-encoder.int8.onnx",
         "models/sherpa-onnx-whisper-small/small-decoder.int8.onnx"),
    )
    num_threads: int = 2
    max_seconds: float = 30.0      # Whisper penceresi 30 sn; ilk 30 sn konuşma yeter
    min_supported: float = 0.5     # desteklenen dillerin toplam olasılığı bunun altındaysa ret
    min_confidence: float = 0.8    # kısıtlı seçimde güven bunun altındaysa "uncertain"


@dataclass(frozen=True)
class VadConfig:
    model: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    min_silence_duration: float = 0.25
    min_speech_duration: float = 0.25
    max_speech_duration: float = 20.0  # Whisper 30 sn'den uzun parçayı kesiyor; 20 güvenli
    # Silero parça sonunu erken kesiyor: Türkçede "mı?", "alın" gibi son kelimeler
    # kayboldu (12 Ekim). Her parça iki yandan bu kadar genişletilir (komşuya taşmadan).
    pad: float = 0.2
    sample_rate: int = 16000


@dataclass(frozen=True)
class DiarizerConfig:
    """Konuşmacı ayrıştırma.

    kind="cluster" (varsayılan): VAD parçaları -> pencere -> CAM++ imzası -> kümeleme.
    kind="pyannote": pyannote segmentation-3.0 + kümeleme (eski yol; kısa konuşma
    sıralarında Vietnamca sesleri ayıramadı, bkz. cluster_diarizer.py).
    """

    kind: str = "cluster"
    # Kümeleme yöntemi: "spectral" (NME-SC: kişi sayısını özdeğer boşluğundan bulur) ya da
    # "ahc" (sabit eşikle birleştirme). 13 Ekim, VietMed'den 9 gerçek görüşme diyaloğu:
    #   ahc 0,70: kişi sayısı 4/9, DER %33,5 | ahc 0,55: 7/9, %17,1 ama tek kişilik 63/70
    #   spectral (birleştirme 0,35): 7/9, DER %17,1, tek kişilik 66/70; sayı verilince %15,9
    # Aynı kaydın iki kişisi arası uzaklık 0,38-0,58, telefondaki tek kişinin kendi içi ~0,3:
    # tek bir sabit eşik ikisine birden uymuyor; spektral yöntem buna daha dayanıklı.
    method: str = "spectral"
    merge_distance: float = 0.35  # spectral: merkezleri bundan yakın kümeler aynı kişi (ölçüm: olcumler.md)
    embedding_model: str = "models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
    segmentation_model: str = "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    # konuşmacı sayısı: 0 = sesten bul (istekte verilirse o kullanılır)
    num_speakers: int = 0
    # kosinüs uzaklığı eşiği. Gerçek sesle ayarlandı (13 Ekim, CAM++, VietMed'den 56 diyalog,
    # telefonla bağlanan hastalar dahil): 0,65 -> sayı 38/56 | 0,70 -> 47/56 | 0,75 -> 29/56.
    # Temiz sentetik seste 0,65 daha iyi (25/28, 0,70'te 22/28); gerçek ses esas alındı.
    cluster_threshold: float = 0.70
    # pencere 1,0 sn / kayma 0,4 sn: sert montaj geçişlerini ve hızlı diyalogları yakalar
    window: float = 1.0          # sn: imza penceresi
    hop: float = 0.4             # sn: pencere kayması (konuşmacı değişimi çözünürlüğü)
    min_window: float = 0.4      # sn: bundan kısa parçadan imza çıkarılmaz
    # Toplamı max(1,5 sn, konuşmanın %8'i)'nden az olan "konuşmacı" en yakın kişiye katılır
    # (gülme, gürültü, müzik ayrı kişi sanılmasın). VietMed 50 tek kişilik dosya + 28
    # sentetik diyalog (12 Ekim):  1 sn/%0 -> 34/50, 27/28 | 1-3 sn/%8-10 -> 48/50, 28/28.
    # Oran asıl işi yapıyor; 3 sn mutlak alt sınır kısa mesajlarda gerçek kişileri
    # (13 sn'lik seste 2 sn konuşan) siliyordu. Bedeli: konuşmanın %8'inden azını konuşan
    # kişi ayrı çıkmaz; sayı biliniyorsa istekte num_speakers verilince kural uygulanmaz.
    min_speaker_duration: float = 1.5
    min_speaker_share: float = 0.08
    # Az konuşan küme en yakın kişiden bu kosinüs uzaklığından daha uzaksa birleştirilmez
    # (projedeki "benzerlik koruması", benzerlik 0,5 = uzaklık 0,5). 0 = kapalı. Ölçüme göre.
    small_keep_distance: float = 0.0
    min_duration_on: float = 0.3   # yalnızca pyannote
    min_duration_off: float = 0.5  # yalnızca pyannote
    num_threads: int = 2


@dataclass(frozen=True)
class FusionConfig:
    """Kelimeleri konuşmacılara atama ayarları."""

    # kelimenin orta noktası iki konuşmacı aralığına birden düşüyorsa:
    # baskın konuşmacının örtüşme payı bu oranın altındaysa BELIRSIZ
    min_dominance: float = 0.6
    # orta nokta hiçbir aralığa düşmüyorsa en yakın aralığa bu mesafeye kadar atanır (sn)
    max_gap: float = 0.5
    # aynı konuşmacının arasında bu kadar sessizlik varsa yeni satır başlar (sn)
    new_utterance_gap: float = 1.5
    # satır içinde bu kadar duraklama cümle sınırı sayılır (temiz transkriptte nokta)
    sentence_gap: float = 0.6
    # tek satır en fazla bu kadar sürer (sn); uzun konuşma en uzun duraklamadan bölünür.
    # 13 Ekim arayüz denemesi: bir doktorun 92 sn'si tek satırdı (okuması ve altyazısı zor).
    # 0 = bölme yok.
    max_utterance: float = 25.0
    # Araya sıkışmış kısa konuşmacı (A, kısa B, A): B bu süreden (sn) kısaysa A'ya verilir
    # (projedeki SemanticRefiner "sandwich" düzeltmesi). 0 = kapalı (varsayılan).
    # 14 Ekim, 9 Vietnamca diyalog: DER %10,64 -> 1,0 sn %10,44 | 1,5 sn %10,14 | 2,0 sn %10,07.
    # Kapalı bırakıldı: bu diyaloglar kesitlerden kurulduğu için içinde gerçek kısa araya
    # girme ("ừ", "evet", "hı hı") yok; düzeltme tam da onları siler ve bu sette ölçülemez.
    # Gerçek, kesintisiz görüşme kaydıyla ölçülünce açılabilir (FUSION_SANDWICH_MAX).
    sandwich_max: float = 0.0
    # kelime güveni bunun altındaysa satırda işaretlenir (Zipformer: parça olasılıkları,
    # faster-whisper: kelime olasılığı); düzeltme yapan kişi önce bunlara bakar
    low_conf: float = 0.5
    # Ardışık kekeleme ve hece tekrarlarını (örn. 'thì thì', 'các các các') temizle
    clean_repeats: bool = False
    # Cümle başı büyük harf ve sonuna noktalama (. / ?) ekleme
    punctuate: bool = False
    # Stüdyo/diyalog konuşmacı rollerini çözümleme (örn. [Sunucu - Erkek], [Katılımcı 1 - Kadın])
    detect_roles: bool = False


@dataclass(frozen=True)
class ProfileConfig:
    """İşlem profilleri (fine-tuning projesindeki PIPELINE_PROFILE).

    full: her adım çalışır (varsayılan).
    feedback: kısa ve çok sayıda gelen geri bildirim mesajları için. Ses min_diarize_sec'ten
      kısaysa konuşmacı ayırma atlanır (tek kişi kabul edilir); Whisper beam 1 (açgözlü çözme).
      Zipformer/Omnilingual zaten hızlı olduğu için onlarda değişen bir şey yok.
    Projede ayrıca SNR düşükse gürültü giderme (DeepFilterNet) vardı: PyTorch ister ve
    internetsiz CPU kurulumuna uymaz; alınmadı (ADR 0006).
    """

    default: str = "full"
    min_diarize_sec: float = 10.0


@dataclass(frozen=True)
class SafetyNetConfig:
    """Yazıya dökülmemiş konuşma (fine-tuning projesindeki güvenlik ağı + kapsama ölçüsü).

    VAD'ın konuşma dediği ama hiç kelime çıkmayan, min_gap'ten uzun aralıklar bulunur;
    motor destekliyorsa (Whisper) kenar payıyla yeniden çözülür. Zipformer/Omnilingual'e zaten
    sesin tamamı verildiği için onlarda yalnızca kapsama ölçülür.
    """

    enabled: bool = True
    min_gap: float = 2.0
    margin: float = 0.3


@dataclass(frozen=True)
class PipelineConfig:
    languages: dict[str, LanguageProfile]
    lid: LidConfig | None = field(default_factory=LidConfig)  # None: dil istekte zorunlu
    vad: VadConfig = field(default_factory=VadConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    diarizer: DiarizerConfig = field(default_factory=DiarizerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    safety: SafetyNetConfig = field(default_factory=SafetyNetConfig)
