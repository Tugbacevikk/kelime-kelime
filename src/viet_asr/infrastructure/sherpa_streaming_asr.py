"""Gerçek akışlı (streaming) transducer: ses geldikçe kelime üretir (sherpa-onnx OnlineRecognizer).

Parça beklemez: model 0,64 sn'lik parçalarla çalışır (seda-v0.1), canlı akışta metin konuşma sürerken
büyür. Cümle sonunu modelin kendi bitiş (endpoint) kuralı bulur; uzun konuşmada `max_utterance`
(varsayılan 12 sn) dolunca zorla kapatılır, böylece düzeltme (Whisper) penceresi sınırlı kalır.

Ölçüm (23 Ekim, 120 sn telefon bandı kayıt, CPU 4 iş parçacığı): süre/ses 0,038, WER %22,5 (n=1).
Bkz. docs/olcumler.md.
"""
from __future__ import annotations

import os

import numpy as np
import sherpa_onnx

from viet_asr.application.config import AsrParams, ModelRef
from viet_asr.application.ports import TextNormalizer
from viet_asr.domain import VadSegment, Word
from viet_asr.infrastructure.sherpa_asr import spread_words

SR = 16000
TAIL_SILENCE_SEC = 1.0  # model kartı: son parçanın çözülmesi için sona 1 sn sessizlik eklenir


def _text_of(result) -> str:
    return (result if isinstance(result, str) else getattr(result, "text", "")).strip()


class StreamHandle:
    """Tek bir canlı akışın tanıma durumu (tanıyıcı modeli akışlar arasında paylaşılır)."""

    def __init__(self, recognizer):
        self.rec = recognizer
        self.stream = recognizer.create_stream()

    def _decode(self) -> None:
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)

    def accept(self, frame: np.ndarray) -> None:
        self.stream.accept_waveform(SR, np.asarray(frame, np.float32))
        self._decode()

    def text(self) -> str:
        return _text_of(self.rec.get_result(self.stream))

    def is_endpoint(self) -> bool:
        return bool(self.rec.is_endpoint(self.stream))

    def reset(self) -> None:
        self.rec.reset(self.stream)

    def finish(self) -> str:
        """Akış bitti: sona sessizlik ekleyip son metni alır."""
        self.stream.accept_waveform(SR, np.zeros(int(TAIL_SILENCE_SEC * SR), np.float32))
        self.stream.input_finished()
        self._decode()
        return self.text()


class SherpaStreamingTransducerAsr:
    """Canlı akış için akışlı model. `streaming = True`: StreamTranscriber ses geldikçe `open_stream()`
    ile besler; VAD parçalama yok. `transcribe()` (dosya, Whisper düzeltmesi dışı) da çalışır."""

    segmentation = "full"
    streaming = True
    device = "cpu"

    def __init__(self, model: ModelRef, params: AsrParams, normalizer: TextNormalizer):
        self.name = model.name
        self.params = params
        self.normalizer = normalizer
        self.max_utterance = float(os.environ.get("STREAM_MAX_UTTERANCE_SEC", "12"))
        # cümle sonu: konuşmadan sonra bu kadar sessizlik (sn). 0,8 sn: kısa duraklamada bölmez.
        silence = float(os.environ.get("STREAM_ENDPOINT_SILENCE", "0.8"))
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=model.path("tokens"), encoder=model.path("encoder"),
            decoder=model.path("decoder"), joiner=model.path("joiner"),
            num_threads=params.num_threads, sample_rate=SR, feature_dim=80,
            decoding_method="modified_beam_search", max_active_paths=4,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=silence * 3,  # hiç konuşma çözülmediyse uzun bekle
            rule2_min_trailing_silence=silence,      # konuşma çözüldükten sonra sessizlik
            rule3_min_utterance_length=self.max_utterance,
            provider="cpu")

    def open_stream(self) -> StreamHandle:
        return StreamHandle(self.recognizer)

    def words_for(self, text: str, start: float, end: float) -> list[Word]:
        """Akışlı modelden gelen metni parçanın süresine yayar (kelime zamanı yok, harf sayısıyla orantılı)."""
        return spread_words(text, start, end, self.normalizer)

    def transcribe(self, segments: list[VadSegment]) -> list[Word]:
        out: list[Word] = []
        for seg in segments:
            h = self.open_stream()
            h.accept(seg.samples)
            out += self.words_for(h.finish(), seg.start, seg.end)
        return out
