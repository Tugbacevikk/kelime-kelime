"""Modelleri models/ klasörüne indirir, açar, bpe.vocab ve MANIFEST.json üretir.

    python scripts/download_models.py                          # varsayılanlar: vi=zipformer-68m, tr=faster-whisper-medium
    python scripts/download_models.py --vi zipformer-30m --tr omnilingual-300m
    python scripts/download_models.py --vi all --tr all        # tüm adaylar (karşılaştırma için)
    python scripts/download_models.py --tr none                # yalnızca Vietnamca
    python scripts/download_models.py --hizli seda-v0.1        # canlı akışın anlık modeli (Türkçe, akışlı)

Her zaman inenler: Silero VAD, pyannote segmentation, WeSpeaker, Whisper-tiny ve -small (dil tespiti).
İnternetsiz (air-gapped) sunucu için: bunu internetli makinede çalıştırıp models/
klasörünü olduğu gibi taşıyın. MANIFEST.json içindeki sha256 değerleri ile kontrol edin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viet_asr.infrastructure.catalog import (  # noqa: E402
    ASR_PACKS, AUX_PACKS, BASE, DEFAULT_MODELS, MODEL_CHOICES, Pack, is_folder_model,
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(pack: Pack, dest: Path) -> Path:
    out = dest / pack.target
    if out.exists():
        print(f"var, atlanıyor: {pack.target}")
        return out
    if pack.hf_repo:
        from huggingface_hub import snapshot_download  # faster-whisper ile gelir

        print(f"indiriliyor: {pack.hf_repo}")
        snapshot_download(pack.hf_repo, revision=pack.hf_revision, local_dir=out,
                          allow_patterns=list(pack.files.values()))
        return out
    name = pack.url_path.rsplit("/", 1)[1]
    tmp = dest / name
    print(f"indiriliyor: {name}")
    urllib.request.urlretrieve(f"{BASE}/{pack.url_path}", tmp)
    if name.endswith(".tar.bz2"):
        with tarfile.open(tmp, "r:bz2") as t:
            extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            t.extractall(dest, **extra)
        tmp.unlink()
    if pack.keep and out.is_dir():
        for f in out.iterdir():
            if f.is_file() and f.name not in pack.keep:
                f.unlink()  # ör. fp32 kopyalar: yalnızca int8 kullanılıyor
    return out


def make_bpe_vocab(model_dir: Path) -> None:
    """Hotword için gerekli bpe.vocab (yoksa sherpa-onnx hotword'leri sessizce yok sayar)."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "bpe.model"))
    with (model_dir / "bpe.vocab").open("w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")


def choose(arg: str, lang: str) -> list[str]:
    if arg == "none":
        return []
    if arg == "all":
        return MODEL_CHOICES[lang]
    return [arg]


def _env_model(lang: str) -> str:
    """.env'deki VI_MODEL / TR_MODEL (baslat.bat ile aynı seçim), yoksa varsayılan."""
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    v = os.environ.get(f"{lang.upper()}_MODEL")
    # klasör yolu verilmişse (ince ayarlı model) indirilecek bir şey yok; yedek olarak varsayılan iner
    return DEFAULT_MODELS[lang] if not v or is_folder_model(v) else v


def _env_fast() -> str:
    """Canlı akışta iki aşamalı mod açıksa (.env: STREAM_TWO_STAGE=1) anlık model STREAM_FAST_MODEL; değilse yok."""
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if os.environ.get("STREAM_TWO_STAGE", "0").lower() not in ("1", "true", "evet"):
        return "none"
    return os.environ.get("STREAM_FAST_MODEL", "omnilingual-300m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vi", choices=[*MODEL_CHOICES["vi"], "all", "none"], default=_env_model("vi"))
    p.add_argument("--tr", choices=[*MODEL_CHOICES["tr"], "all", "none"], default=_env_model("tr"))
    p.add_argument("--hizli", choices=[*MODEL_CHOICES["tr"], "none"], default=_env_fast(),
                   help="canlı akışın anlık modeli (Türkçe): seda-v0.1 gibi; .env'deki STREAM_FAST_MODEL varsayılan")
    p.add_argument("--dest", default="models")
    a = p.parse_args()
    dest = Path(a.dest)
    dest.mkdir(exist_ok=True)

    packs = dict(AUX_PACKS)
    for key in choose(a.vi, "vi") + choose(a.tr, "tr") + choose(a.hizli, "tr"):
        packs[key] = ASR_PACKS[key]

    mpath = dest / "MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    for key, pack in packs.items():
        if key in manifest and (dest / pack.target).exists():
            continue  # inmiş ve kayıtlı: yeniden özet hesaplamaya gerek yok (her başlatmada hızlı)
        path = fetch(pack, dest)
        if pack.needs_bpe_vocab and not (path / "bpe.vocab").exists():
            make_bpe_vocab(path)
        if path.is_file():
            files = [path]
        elif pack.hf_repo:
            files = [path / f for f in pack.files.values()]
        else:
            files = sorted(path.rglob("*.onnx"))
        url = f"hf://{pack.hf_repo}" if pack.hf_repo else f"{BASE}/{pack.url_path}"
        manifest[key] = {"url": url, "license": pack.license,
                         "sha256": {f.relative_to(dest).as_posix(): sha256(f) for f in files}}
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Tamam. {mpath} yazıldı.")


if __name__ == "__main__":
    main()
