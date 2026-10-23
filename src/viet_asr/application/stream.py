"""Canlı akışın gerçek zamanlı çözümü: ses parçaları -> VAD -> ASR -> anında metin.

    kaynak.frames()  ─►  StreamingVad.feed  ─► biten konuşma parçası ─► ASR ─► on_segment(final)
                                └─ konuşma sürerken her `interim_every` sn ─► ASR ─► on_segment(ara)

- Parça sınırı konuşmadaki duraklama (cümle ortasından kesmez); uzun konuşmada en geç VAD'ın
  max_speech süresinde (10 sn) parça kapanır: gecikmenin üst sınırı.
- Sesin tamamına bakan motorlarda (Zipformer, segmentation="full") parçalar BİTİŞİK: VAD
  yalnızca kesim yerini seçer, aradaki ses atılmaz. Dosya hattındaki gibi: VAD müzik altındaki ya
  da kısık konuşmayı kaçırıyor. 16 Ekim, 2 VietMed kaydı RTP ile: yalnızca VAD parçaları
  çözülünce WER %13,8 / %21,0 (çoğu silinen kelime), aynı kaydın dosyadan çözümü %4,5 / %14,0.
- Ara sonuç (final=False) yalnızca hızlı motorlarda (Zipformer): Whisper'da her ara çözüm
  saniyeler sürer, canlı akışı geride bırakır.
- Konuşmacı ayırma burada yok: telefon santralleri her tarafı ayrı RTP akışı olarak verir
  (taraf = akış). Tek akışta birden çok kişi varsa kesin sonuç görüşme bitince, kaydedilen ses
  normal hattan (konuşmacı ayırma + temiz transkript) geçirilerek alınır (interface/streams.py).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from viet_asr.domain import StreamSegment, VadSegment, Word

from .ports import AsrEngine, AudioStream, StreamingVad

# 2 sn / 10 sn bekleme ölçümde fark yaratmadı: medium canlıya yetişemediği için kuyruk hep dolu ve gruplar
# bekleme süresinden değil birikmeden oluşuyor (docs/olcumler.md). Eski davranış korundu.
REFINE_IDLE_SEC = 2.0     # yeni parça gelmezse toplanan düzeltme grubu bu süreden sonra çözülür
REFINE_FLUSH_SPAN = 29.5  # grup bu kadar sese ulaşınca beklemeden çözülür (Whisper penceresi)
REFINE_MAX_GAP = 3.0     # iki parça arası bu kadar sessizlik varsa grup bölünür
REFINE_JOIN_SEC = 60.0    # akış bitince bekleyen düzeltmeler için azami bekleme
REFINE_JOIN_STOP_SEC = 3.0  # iptalde bekleme

log = logging.getLogger(__name__)
SR = 16000


@dataclass
class StreamSummary:
    duration: float = 0.0
    segments: int = 0
    words: int = 0
    max_latency: float = 0.0
    latencies: list[float] = field(default_factory=list)
    kuyruk_beklemeler: list[float] = field(default_factory=list)
    max_kuyruk_bekleme: float = 0.0
    toplam_gecikmeler: list[float] = field(default_factory=list)
    max_toplam_gecikme: float = 0.0
    duzeltmeler: int = 0
    duzeltme_pencere: int = 0  # medium'a giden çözme çağrısı sayısı (az = iyi paketlenmiş)

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, 95))

    @property
    def p95_toplam_gecikme(self) -> float:
        if not self.toplam_gecikmeler:
            return 0.0
        return float(np.percentile(self.toplam_gecikmeler, 95))


class StreamTranscriber:
    def __init__(self, vad_factory: Callable[[], StreamingVad], low_conf: float = 0.5,
                 interim_every: float = 1.5, min_interim: float = 1.0, max_chunk: float = 8.0,
                 min_chunk: float = 5.0):
        self.vad_factory = vad_factory
        self.low_conf = low_conf
        self.interim_every, self.min_interim = interim_every, min_interim
        # bitişik modda VAD kesim vermezse (duraklamasız uzun konuşma) en sessiz yerden zorla kes.
        # 4 VietMed kaydında çevrimdışı benzetim: 15 sn -> WER %11,0, kesin metin en geç 7,1 sn
        # sonra; 8 sn -> WER %10,8, en geç 3,0 sn (kesim parçanın son %40'ında aranır).
        self.max_chunk = max_chunk
        # bitişik modda parça en az bu kadar (dosya hattındaki EnergyChunker min_len): kısa parçada
        # bağlam az, dolgu sesleri daha çok yazılıyor. Bu arada ekranda ara sonuç görünür.
        self.min_chunk = min_chunk

    def run(self, source: AudioStream, asr: AsrEngine, on_segment: Callable[[StreamSegment], None],
            record: Callable[[np.ndarray], None] | None = None, interim: bool = True,
            stop: threading.Event | None = None,
            refiner: AsrEngine | None = None,
            on_refine: Callable[[StreamSegment], None] | None = None) -> StreamSummary:
        native = bool(getattr(asr, "streaming", False))  # gerçek akışlı model: ses geldikçe beslenir, VAD yok
        contiguous = getattr(asr, "segmentation", "vad") == "full"
        try:  # bitişik modda VAD'ın zorla kesmesi (kelime ortası) yerine en sessiz yerden keseriz
            vad = self.vad_factory(max_speech=self.max_chunk * 2) if contiguous else self.vad_factory()
        except TypeError:
            vad = self.vad_factory()
        summary = StreamSummary()
        arrived: list[tuple[int, float, float]] = []  # (o ana kadar gelen örnek, pop saati, giriş saati)
        total, index, last_interim = 0, 0, 0.0

        refine_q: queue.Queue[tuple[StreamSegment, VadSegment, float] | None] | None = None
        emit_lock = threading.Lock()  # düzeltme iş parçacığı ile ana döngü aynı anda olay yayınlamasın
        _on_segment, _on_refine = on_segment, on_refine

        def on_segment(seg: StreamSegment) -> None:
            with emit_lock:
                _on_segment(seg)
        refine_t: threading.Thread | None = None

        if refiner is not None:
            refine_q = queue.Queue()

            def _process_refine_batch(batch: list[tuple[StreamSegment, VadSegment, float]]) -> None:
                if not batch:
                    return
                t0 = min(s.start for _, s, _ in batch)
                shifted = [VadSegment(start=max(0.0, s.start - t0), samples=s.samples) for _, s, _ in batch]
                summary.duzeltme_pencere += 1
                t_start = time.monotonic()
                try:
                    words = refiner.transcribe(shifted)
                except Exception as e:
                    log.warning("düzeltme motoru başarısız (%s), hızlı metinler korunuyor", e)
                    return
                words = [Word(text=w.text, start=w.start + t0, end=w.end + t0, conf=w.conf) for w in words]
                seg_words: list[list[Word]] = [[] for _ in batch]
                for w in words:
                    mid = (w.start + w.end) / 2.0
                    best_i = 0
                    min_dist = float("inf")
                    for i, (_, s, _) in enumerate(batch):
                        if s.start <= mid <= s.end:
                            best_i = i
                            break
                        dist = min(abs(mid - s.start), abs(mid - s.end))
                        if dist < min_dist:
                            min_dist = dist
                            best_i = i
                    seg_words[best_i].append(w)

                t_now = time.monotonic()
                for (st_seg, vad_seg, t_in), ws in zip(batch, seg_words):
                    txt = " ".join(w.text for w in ws).strip()
                    if not txt:
                        txt = st_seg.text
                    confs = [w.conf for w in ws if w.conf is not None]
                    avg_conf = round(sum(confs) / len(confs), 3) if confs else st_seg.conf
                    ref_seg = StreamSegment(
                        index=st_seg.index,
                        start=st_seg.start,
                        end=st_seg.end,
                        text=txt,
                        final=True,
                        conf=avg_conf,
                        low_conf=(),
                        latency=round(max(0.0, t_now - t_start), 3),  # düzeltme motorunun bu pencereyi çözme süresi
                        kuyruk_bekleme_sn=st_seg.kuyruk_bekleme_sn,
                        toplam_gecikme_sn=round(max(0.0, t_now - t_in), 3),
                        duzeltildi=True,
                    )
                    summary.duzeltmeler += 1
                    if _on_refine is not None:
                        with emit_lock:
                            _on_refine(ref_seg)
                    else:
                        on_segment(ref_seg)

            def _refine_loop() -> None:
                batch: list[tuple[StreamSegment, VadSegment, float]] = []
                batch_start = 0.0
                last_end = 0.0
                while True:
                    timeout = REFINE_IDLE_SEC if batch else None
                    try:
                        item = refine_q.get(timeout=timeout)
                    except queue.Empty:
                        if batch:
                            _process_refine_batch(batch)
                            batch = []
                        continue

                    if item is None:
                        if batch:
                            _process_refine_batch(batch)
                        break

                    st_seg, vad_seg, t_in = item
                    if batch:
                        span = vad_seg.end - batch_start
                        gap = vad_seg.start - last_end
                        if span > REFINE_FLUSH_SPAN or gap > REFINE_MAX_GAP:
                            _process_refine_batch(batch)
                            batch = []

                    if not batch:
                        batch_start = vad_seg.start

                    batch.append((st_seg, vad_seg, t_in))
                    last_end = vad_seg.end
                    if last_end - batch_start >= REFINE_FLUSH_SPAN:
                        _process_refine_batch(batch)
                        batch = []

            refine_t = threading.Thread(target=_refine_loop, daemon=True, name="refine-worker")
            refine_t.start()

        def emit(seg: VadSegment, final: bool) -> None:
            nonlocal index
            words = asr.transcribe([seg]) if len(seg.samples) else []
            end_sample = int(round(seg.end * SR))
            k = bisect_left(arrived, (end_sample, 0.0, 0.0))
            if k < len(arrived):
                _, t_pop, t_in = arrived[k]
            else:
                t_pop = t_in = time.monotonic()
            t_now = time.monotonic()
            latency = max(0.0, t_now - t_pop)
            queue_wait = max(0.0, t_pop - t_in)
            total_latency = max(0.0, t_now - t_in)
            out = _segment(index, seg, words, final, latency, self.low_conf,
                           queue_wait=queue_wait, total_latency=total_latency)
            if final:
                index += 1
                summary.segments += 1
                summary.words += len(words)
                summary.latencies.append(latency)
                summary.max_latency = max(summary.max_latency, latency)
                summary.kuyruk_beklemeler.append(queue_wait)
                summary.max_kuyruk_bekleme = max(summary.max_kuyruk_bekleme, queue_wait)
                summary.toplam_gecikmeler.append(total_latency)
                summary.max_toplam_gecikme = max(summary.max_toplam_gecikme, total_latency)
                if refine_q is not None and len(seg.samples):
                    refine_q.put((out, seg, t_in))
            if out.text or final:
                on_segment(out)

        buf: list[np.ndarray] = []   # bitişik mod: son kesimden bu yana gelen ses
        buf_start = 0                # buf'un akıştaki ilk örneği

        def take(cut: int) -> VadSegment | None:
            """buf'tan [buf_start, cut) aralığını parça yapıp çıkarır."""
            nonlocal buf, buf_start
            if cut <= buf_start:
                return None
            a = np.concatenate(buf) if buf else np.zeros(0, np.float32)
            n = cut - buf_start
            seg = VadSegment(start=buf_start / SR, samples=a[:n])
            buf, buf_start = ([a[n:]] if len(a) > n else []), cut
            return seg

        handle = asr.open_stream() if native else None
        last_text = ""
        level = 0.0      # son konuşma seviyesi (yavaşça sönen en yüksek ses)
        quiet_run = 0    # akışın sonundaki ardışık sessiz örnek sayısı

        def native_emit(final: bool, t_pop: float, t_in: float, text: str | None = None) -> None:
            """Akışlı modelin o ana kadarki metnini ara (final=False) ya da kesin parça olarak yayınlar."""
            nonlocal index, last_text
            text = (handle.text() if text is None else text).strip()
            end_sample = total
            if not final:
                if text and text != last_text and buf:
                    last_text = text
                    seg = VadSegment(start=buf_start / SR, samples=np.concatenate(buf))
                    words = asr.words_for(text, seg.start, seg.end)
                    on_segment(_segment(index, seg, words, False, max(0.0, time.monotonic() - t_pop),
                                        self.low_conf))
                return
            seg = take(end_sample)
            last_text = ""
            handle.reset()
            if seg is None or not text:
                return  # konuşma çözülmedi (sessizlik/gürültü): parça yok
            words = asr.words_for(text, seg.start, seg.end)
            t_now = time.monotonic()
            latency, queue_wait, total_latency = max(0.0, t_now - t_pop), max(0.0, t_pop - t_in), max(0.0, t_now - t_in)
            out = _segment(index, seg, words, True, latency, self.low_conf,
                           queue_wait=queue_wait, total_latency=total_latency)
            index += 1
            summary.segments += 1
            summary.words += len(words)
            summary.latencies.append(latency)
            summary.max_latency = max(summary.max_latency, latency)
            summary.kuyruk_beklemeler.append(queue_wait)
            summary.max_kuyruk_bekleme = max(summary.max_kuyruk_bekleme, queue_wait)
            summary.toplam_gecikmeler.append(total_latency)
            summary.max_toplam_gecikme = max(summary.max_toplam_gecikme, total_latency)
            if refine_q is not None and len(seg.samples):
                refine_q.put((out, seg, t_in))
            on_segment(out)

        try:
            for frame in source.frames():
                if stop is not None and stop.is_set():
                    break
                frame = np.asarray(frame, dtype=np.float32)
                if record is not None:
                    record(frame)
                t_pop = time.monotonic()
                t_in = getattr(source, "last_ingress_time", None)
                if t_in is None:
                    t_in = t_pop
                total += len(frame)
                arrived.append((total, t_pop, t_in))
                if len(arrived) > 4096:
                    del arrived[:2048]
                if contiguous:
                    buf.append(frame)
                if native:
                    handle.accept(frame)
                    rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
                    level = max(level * 0.999, rms)
                    quiet_run = quiet_run + len(frame) if rms < 0.15 * level else 0
                    utt = total - buf_start
                    soft = getattr(asr, "soft_utterance", 0.0)
                    # uzun cümle: kısa bir nefeste kes (kelime ortasından zorla kesmemek için)
                    soft_cut = soft > 0 and utt > soft * SR and quiet_run >= getattr(asr, "soft_pause", 0.25) * SR
                    if handle.is_endpoint() or soft_cut or utt > getattr(asr, "max_utterance", 12.0) * SR:
                        native_emit(True, t_pop, t_in)   # cümle sonu, nefes payı ya da zorla kesim
                        quiet_run = 0
                    elif interim:
                        native_emit(False, t_pop, t_in)
                    continue
                for seg in vad.feed(frame):
                    if contiguous:
                        cut = min(total, int(round(seg.end * SR)))
                        if cut - buf_start < self.min_chunk * SR:
                            continue  # kısa: sonraki duraklamada birlikte kesilir
                        seg = take(cut)
                        if seg is None:
                            continue
                    emit(seg, True)
                    last_interim = total / SR
                if contiguous and total - buf_start > self.max_chunk * SR:
                    emit(take(buf_start + _quietest_cut(np.concatenate(buf))), True)
                    last_interim = total / SR
                if interim and total / SR - last_interim >= self.interim_every:
                    cur = vad.ongoing()
                    if contiguous and cur is not None and buf:
                        cur = VadSegment(start=buf_start / SR, samples=np.concatenate(buf))
                    if cur is not None and cur.end - cur.start >= self.min_interim:
                        emit(cur, False)
                    last_interim = total / SR
            if native:
                native_emit(True, time.monotonic(), time.monotonic(), text=handle.finish())
                tail = []
            else:
                tail = vad.flush()
            if native:
                pass
            elif contiguous:
                seg = take(total)
                if seg is not None and seg.end - seg.start >= 0.3:
                    emit(seg, True)
            else:
                for seg in tail:
                    emit(seg, True)
        finally:
            if refine_q is not None and refine_t is not None:
                refine_q.put(None)
                refine_t.join(timeout=REFINE_JOIN_STOP_SEC if (stop is not None and stop.is_set())
                              else REFINE_JOIN_SEC)
                if refine_t.is_alive():
                    log.warning("düzeltme bitmeden akış kapandı: %d parça düzeltilmedi",
                                max(0, summary.segments - summary.duzeltmeler))
            source.close()
        summary.duration = total / SR
        return summary


def _quietest_cut(a: np.ndarray, frame: float = 0.1) -> int:
    """Zorla kesim: parçanın son %40'ındaki en sessiz 100 ms'nin ortası (kelime ortasından
    kesmez; sona yakın kestiği için kesin metin geç kalmaz)."""
    n = int(frame * SR)
    k = len(a) // n
    if k < 2:
        return len(a)
    rms = np.sqrt((a[:k * n].reshape(k, n) ** 2).mean(axis=1))
    lo = int(k * 0.6)
    i = lo + int(np.argmin(rms[lo:]))
    return i * n + n // 2


def _segment(index: int, seg: VadSegment, words: list[Word], final: bool, latency: float,
             low: float, queue_wait: float = 0.0, total_latency: float = 0.0,
             duzeltildi: bool = False) -> StreamSegment:
    confs = [w.conf for w in words if w.conf is not None]
    return StreamSegment(
        index=index, start=round(words[0].start if words else seg.start, 3),
        end=round(max([w.end for w in words] + [seg.end]) if words else seg.end, 3),
        text=" ".join(w.text for w in words), final=final,
        conf=round(sum(confs) / len(confs), 3) if confs else None,
        low_conf=tuple(i for i, w in enumerate(words) if w.conf is not None and w.conf < low),
        latency=round(latency, 3),
        kuyruk_bekleme_sn=round(queue_wait, 3),
        toplam_gecikme_sn=round(total_latency, 3),
        duzeltildi=duzeltildi)
