"""Model kataloğu: indirme adresleri, klasör adları, dosya rolleri, lisanslar.

Hem scripts/download_models.py hem de CLI buradan okur; model bilgisi tek yerde.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from viet_asr.application.config import AsrParams, LanguageProfile, ModelRef

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class Pack:
    url_path: str           # BASE'e göre
    target: str             # açılınca oluşan klasör ya da dosya adı
    license: str
    kind: str | None = None  # ASR ise adaptör türü
    files: dict[str, str] = field(default_factory=dict)
    needs_bpe_vocab: bool = False
    keep: tuple[str, ...] = ()  # boş değilse açıldıktan sonra yalnızca bu dosyalar kalır
    hf_repo: str | None = None  # sherpa-onnx yerine Hugging Face deposundan (url_path kullanılmaz)
    hf_revision: str | None = None  # sabit sürüm (commit): aynı komut hep aynı dosyaları indirir


ASR_PACKS: dict[str, Pack] = {
    "zipformer-68m": Pack(
        "asr-models/sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2",
        "sherpa-onnx-zipformer-vi-int8-2025-04-20", "Apache-2.0", "zipformer-transducer",
        {"encoder": "encoder-epoch-12-avg-8.int8.onnx", "decoder": "decoder-epoch-12-avg-8.onnx",
         "joiner": "joiner-epoch-12-avg-8.int8.onnx", "tokens": "tokens.txt", "bpe_vocab": "bpe.vocab"},
        needs_bpe_vocab=True),
    "zipformer-30m": Pack(
        "asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2",
        "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        "CC BY-NC-ND 4.0 (ticari kullanım ve ince ayar yasak)", "zipformer-transducer",
        {"encoder": "encoder.int8.onnx", "decoder": "decoder.onnx", "joiner": "joiner.int8.onnx",
         "tokens": "tokens.txt", "bpe_vocab": "bpe.vocab"},
        needs_bpe_vocab=True),
    "whisper-turbo": Pack(
        "asr-models/sherpa-onnx-whisper-turbo.tar.bz2", "sherpa-onnx-whisper-turbo", "MIT",
        "whisper",
        {"encoder": "turbo-encoder.int8.onnx", "decoder": "turbo-decoder.int8.onnx",
         "tokens": "turbo-tokens.txt"}),
    # Fine-tuning projesindeki Türkçe model ailesi, CTranslate2 int8. Gerçek kelime zamanı verir
    # (sherpa-onnx Whisper vermiyor). Varsayılan medium (13 Ekim kararı); small daha hızlı seçenek.
    "faster-whisper-medium": Pack(
        "", "faster-whisper-medium", "MIT", "faster-whisper",
        {"model": "model.bin", "config": "config.json", "tokenizer": "tokenizer.json",
         "vocabulary": "vocabulary.txt"},
        hf_repo="Systran/faster-whisper-medium"),
    "faster-whisper-small": Pack(
        "", "faster-whisper-small", "MIT", "faster-whisper",
        {"model": "model.bin", "config": "config.json", "tokenizer": "tokenizer.json",
         "vocabulary": "vocabulary.txt"},
        hf_repo="Systran/faster-whisper-small"),
    # Türkçe, gerçek akışlı Zipformer2 transducer (66M, Apache-2.0; LM dosyaları CC BY-SA, indirilmez).
    # Yalnızca canlı akışın anlık metni için (STREAM_FAST_MODEL=seda-v0.1). 23 Ekim ölçümü: docs/olcumler.md.
    "seda-v0.1": Pack(
        "", "seda-v0.1", "Apache-2.0", "zipformer-streaming",
        {"encoder": "encoder.int8.onnx", "decoder": "decoder.onnx", "joiner": "joiner.int8.onnx",
         "tokens": "tokens.txt"},
        hf_repo="atasoglu/seda-v0.1"),
    "omnilingual-300m": Pack(
        "asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2",
        "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12", "Apache-2.0",
        "omnilingual-ctc", {"model": "model.int8.onnx", "tokens": "tokens.txt"}),
}

AUX_PACKS: dict[str, Pack] = {
    "silero-vad": Pack("asr-models/silero_vad.onnx", "silero_vad.onnx", "MIT"),
    "pyannote-seg": Pack(
        "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "sherpa-onnx-pyannote-segmentation-3-0", "MIT"),
    "wespeaker": Pack(
        # 'recongition' yazım hatası sherpa-onnx'in kendi adresinde var, düzeltmeyin
        "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
        "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "WeSpeaker (Apache-2.0 kod; VoxCeleb ile eğitilmiş ağırlık lisansı doğrulanacak)"),
    # Konuşmacı imzası. 13 Ekim, VietMed'den kurulan 56 gerçek diyalog (sayı / cümle / RTF):
    #   CAM++ common      47/56  %91,7  0,030   <- seçilen
    #   ERes2Net base     40/56  %85,6  0,049
    #   ERes2NetV2        40/56  %87,1  0,103
    #   TitaNet large     38/56  %85,0  0,055
    #   ResNet293         CPU'da çok yavaş, ölçüm yarıda bırakıldı
    "campplus": Pack(
        # 'recongition' yazım hatası sherpa-onnx'in kendi adresinde var, düzeltmeyin
        "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "3D-Speaker / ModelScope (Apache-2.0 olduğu doğrulanacak)"),
    "whisper-tiny-lid": Pack(
        "asr-models/sherpa-onnx-whisper-tiny.tar.bz2", "sherpa-onnx-whisper-tiny", "MIT",
        keep=("tiny-encoder.int8.onnx", "tiny-decoder.int8.onnx", "tiny-tokens.txt")),
    "whisper-small-lid": Pack(
        "asr-models/sherpa-onnx-whisper-small.tar.bz2", "sherpa-onnx-whisper-small", "MIT",
        keep=("small-encoder.int8.onnx", "small-decoder.int8.onnx", "small-tokens.txt")),
    "lid-test-wavs": Pack(
        "asr-models/spoken-language-identification-test-wavs.tar.bz2",
        "spoken-language-identification-test-wavs", "test sesleri (sherpa-onnx)"),
}

# Her dilin varsayılan modeli. Ölçüm (12 Ekim, 2 çekirdek):
#   Türkçe, 24 sentetik cümle:  whisper-turbo CER %3,0 | omnilingual-300m CER %3,8
#   Türkçe, 60 sn iki konuşmacılı diyalog (23 VAD parçası), tüm hat:
#     whisper-turbo 63,5 sn (ASR 53,7)  | omnilingual-300m 24,9 sn (ASR 12,1)
#   Whisper her parçayı 30 sn'ye doldurup işliyor; kısa konuşma sıralarında
#   maliyet parça sayısıyla artıyor ve kelime zamanı vermiyor. Omnilingual
#   gerçek zaman damgası veriyor, ama "varmı", "birkan" gibi bitişik yazım
#   hataları ve noktalamasız küçük harf çıktı var.
#   -> hız hedefini (1 dk ses <= 1 dk) karşılayan omnilingual-300m.
# Türkçe varsayılanı faster-whisper medium (13 Ekim, kullanıcı kararı). Bu ortamda indirilemediği
# için hızı ÖLÇÜLMEDİ. CPU'da small'dan ~2-3 kat yavaş olması beklenir; önceki hatta small,
# kullanıcının bilgisayarında oran 0,5-0,7 verdi. Ölçüm:
#   python benchmarks/kabul_testi.py --dil tr kayit.wav   (1 dk ses <= 1 dk mı?)
# Oran 1'i aşarsa sırasıyla: TR_MODEL=faster-whisper-small, sonra omnilingual-300m.
DEFAULT_MODELS = {"vi": "zipformer-68m", "tr": "faster-whisper-medium"}
MODEL_CHOICES = {"vi": ["zipformer-68m", "zipformer-30m", "faster-whisper-medium", "faster-whisper-small"],
                 "tr": ["faster-whisper-medium", "faster-whisper-small", "omnilingual-300m",
                        "whisper-turbo", "seda-v0.1"]}

# Bellek tahmini (GB): (model ağırlığı — bir süreçte bir kez, iş başına çalışma belleği).
# Ölçülen: Zipformer-68M 10 sn'lik gruplarla tüm süreç 0,46 GB (12 Ekim). Diğerleri model
# boyutundan (int8) ve kütüphane belgelerinden tahmin; faster-whisper'da iş başına değer, aynı
# anda çalışan her işçinin modelin kopyasını tutma olasılığına göre ihtiyatlı seçildi.
MODEL_MEMORY_GB: dict[str, tuple[float, float]] = {
    "zipformer-68m": (0.2, 0.3),
    "zipformer-30m": (0.1, 0.3),
    "omnilingual-300m": (0.4, 0.5),
    "seda-v0.1": (0.2, 0.2),  # ölçülmedi: model 79 MB, kartında ~170 MB RAM yazıyor
    "whisper-turbo": (0.9, 0.8),
    "faster-whisper-small": (0.3, 0.5),
    "faster-whisper-medium": (0.8, 1.0),
}
# her süreçte bir kez: VAD + konuşmacı imzası + dil tespiti (Whisper tiny/small) + Python
AUX_MEMORY_GB = 0.9


def memory_estimate(model: str) -> tuple[float, float]:
    """Katalogda olmayan (klasör yolu verilen, ör. ince ayarlı Zipformer) model: Zipformer gibi."""
    return MODEL_MEMORY_GB.get(model, MODEL_MEMORY_GB["zipformer-68m"])


# Whisper'ın sessizlik/gürültüde uydurduğu, gözlenmiş cümleler
# (bir parçanın metni bunlardan biriyse atılır; Live-stream'de canlı denemelerde görülenler eklendi)
DROP_PHRASES = {"tr": (
    "İzlediğiniz için teşekkür ederim.", "İzlediğiniz için teşekkürler.", "Altyazı M.K.",
    "Altyazı", "Abone olmayı unutmayın.", "Abone olun.", "Videoyu beğenmeyi unutmayın.",
    "Bir sonraki videoda görüşmek üzere.", "Müzik", "Thanks for watching!",
    "Subtitles by the Amara.org community",
)}


def lid_stages(models_dir: str | Path) -> tuple[tuple[str, str], ...]:
    d = Path(models_dir)
    return tuple((str(d / f"sherpa-onnx-whisper-{s}/{s}-encoder.int8.onnx"),
                  str(d / f"sherpa-onnx-whisper-{s}/{s}-decoder.int8.onnx")) for s in ("tiny", "small"))


def model_ref(key: str, models_dir: str | Path) -> ModelRef:
    p = ASR_PACKS[key]
    return ModelRef(name=key, kind=p.kind, folder=Path(models_dir) / p.target,
                    files=dict(p.files), license=p.license)


def _pick(folder: Path, pattern: str, prefer_int8: bool) -> str:
    found = sorted(folder.glob(pattern))
    int8 = [f for f in found if f.name.endswith(".int8.onnx")]
    fp32 = [f for f in found if not f.name.endswith(".int8.onnx")]
    order = (int8 + fp32) if prefer_int8 else (fp32 + int8)
    if not order:
        raise FileNotFoundError(f"{folder} içinde {pattern} yok")
    return order[0].name


def folder_ref(folder: str | Path) -> ModelRef:
    """Katalogda olmayan, klasörü verilen Zipformer (ör. ince ayarlı model; finetune/ çıktısı).
    icefall export-onnx.py adlandırması: encoder/joiner int8, decoder fp32 (sherpa-onnx önerisi)."""
    d = Path(folder)
    files = {"encoder": _pick(d, "encoder*.onnx", True), "decoder": _pick(d, "decoder*.onnx", False),
             "joiner": _pick(d, "joiner*.onnx", True), "tokens": "tokens.txt"}
    if (d / "bpe.vocab").exists():
        files["bpe_vocab"] = "bpe.vocab"
    lic = (d / "LICENSE").read_text(encoding="utf-8").strip()[:200] if (d / "LICENSE").exists() else ""
    return ModelRef(name=d.name, kind="zipformer-transducer", folder=d, files=files, license=lic)


def is_folder_model(model: str | None) -> bool:
    return bool(model) and ("/" in model or "\\" in model or Path(model).is_dir())


def language_profile(code: str, models_dir: str | Path, model: str | None = None,
                     **params) -> LanguageProfile:
    key = model or DEFAULT_MODELS[code]
    params.setdefault("drop_phrases", DROP_PHRASES.get(code, ()))
    if is_folder_model(key):  # VI_MODEL=models/vi-ince-ayar-2026-11 gibi
        ref = folder_ref(key)
        return LanguageProfile(code=code, asr_model=ref, asr=AsrParams(**params), segmentation="full")
    if key not in MODEL_CHOICES.get(code, []):
        raise ValueError(f"'{key}' modeli '{code}' dili için tanımlı değil: {MODEL_CHOICES.get(code)} "
                         "(ya da model klasörünün yolunu verin)")
    ref = model_ref(key, models_dir)
    # Whisper sessizlik/müzikte metin uyduruyor: ona yalnızca VAD parçaları
    seg = "vad" if ref.kind in ("whisper", "faster-whisper") else "full"
    return LanguageProfile(code=code, asr_model=ref, asr=AsrParams(**params), segmentation=seg)
