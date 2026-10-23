"""Canlı akış oturumları: kaynağı aç, gerçek zamanlı çöz, kaydet, bitince tam analize ver.

Kaynak yazımı (POST /streams "kaynak", CLI --kaynak):
    rtp://0.0.0.0:5004        UDP portunda RTP dinle (G.711 PCMU/PCMA, L16; bizim alıcı)
    rtp://0.0.0.0:0           boş port seçilir, yanıtta "port" döner
    /yol/cagri.sdp            RTP + SDP (Opus, G.722...; ffmpeg)
    push                      sesi karşı servis parça parça yollar (POST/WS /streams/{id}/audio)
    http(s)://.../akim.ogg    OGG/MP3/WAV yayını (ffmpeg); rtsp://, srt://, udp:// da olur
    /yol/kayit.wav            dosyayı gerçek zaman hızında oynat (deneme; ALLOW_LOCAL_PATHS=1)

İki geçiş: canlı metin anında (StreamTranscriber), akış bitince kaydedilen ses normal işe
dönüşür (konuşmacı ayırma, güvenlik ağı, temiz transkript). İşin kimliği oturumda "son_is".
"""
from __future__ import annotations

import logging
import queue as queue_mod
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import soundfile as sf

from viet_asr.application.config import VadConfig
from viet_asr.application.stream import StreamTranscriber
from viet_asr.domain import AudioJob, StreamSegment
from viet_asr.infrastructure.ffmpeg_source import FfmpegSource
from viet_asr.infrastructure.push_source import PushSource
from viet_asr.infrastructure.rtp import RtpSource
from viet_asr.infrastructure.sherpa_vad import SileroStreamingVad

log = logging.getLogger(__name__)

URL_SCHEMES = ("http", "https", "rtsp", "srt", "udp", "rtmp")
# ara sonuç yalnızca hızlı motorlarda (Whisper ara çözümleri canlı akışı geride bırakır)
FAST_ENGINES = ("zipformer", "omnilingual", "seda")


class StreamError(ValueError):
    pass


@dataclass
class StreamSpec:
    kaynak: str
    dil: str = "vi"
    codec: str = "auto"
    clock_rate: int | None = None
    son_analiz: bool = True
    num_speakers: int | None = None
    callback_url: str | None = None
    idle_timeout: float = 5.0
    ses_hizi: int = 16000    # push kaynağında gönderilen PCM16'nın örnekleme hızı
    canli_url: str | None = None  # canlı parçalar buraya POST edilir (imzalı); callback_url bitişteki işe aittir
    sdp: str | None = None   # SDP metni (santralin verdiği); kaynak yerine ya da rtp:// ile birlikte
    iki_asama: bool | None = None  # True: hızlı model + arka plan Whisper düzeltmesi; None: STREAM_TWO_STAGE
    duzeltme: bool = True    # False: iki aşamalıda yalnız hızlı model (Whisper düzeltmesi yok), ölçüm için


def source_from_sdp(text: str, spec: StreamSpec, work_dir: Path | None = None):
    """SDP'ye göre alıcı: PCMU/PCMA/L16 bizim RTP alıcımız, diğerleri (Opus, G.722) ffmpeg.
    kaynak rtp://adres:port verilmişse dinlenecek adres/port oradan (port 0 = boş port)."""
    from viet_asr.infrastructure import sdp as sdp_mod

    try:
        media = sdp_mod.parse(text)
    except sdp_mod.SdpError as e:
        raise StreamError(f"SDP: {e}") from e
    u = urlparse(spec.kaynak.strip()) if spec.kaynak and spec.kaynak.startswith("rtp://") else None
    host = (u.hostname if u else None) or "0.0.0.0"  # nosec B104 (RTP dinleme adresi varsayılanı)
    port = u.port if (u and u.port is not None) else media.port
    native = {f.pt: (f.native, f.clock_rate, f.channels) for f in media.formats if f.native}
    if native:
        return RtpSource(host, port, "auto", None, idle_timeout=spec.idle_timeout, formats=native)
    if u and u.port is not None and u.port != media.port:
        raise StreamError("ffmpeg ile çözülen biçimlerde (Opus, G.722) port SDP'deki m= portu olmalı")
    # ffmpeg SDP'yi dosyadan okur; geçici dosya akış bitince silinir
    import tempfile

    d = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    d.mkdir(parents=True, exist_ok=True)
    f = tempfile.NamedTemporaryFile("w", suffix=".sdp", dir=d, delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    src = FfmpegSource(f.name, idle_timeout=spec.idle_timeout, cleanup=f.name)
    src.port = media.port
    return src


def bind_rtp(host: str, port: int, spec: StreamSpec, rtp_ports: tuple[int, int] | None):
    """port 0 ve aralık tanımlıysa (STREAM_RTP_PORTS) aralıktaki ilk boş portu alır: güvenlik duvarı
    ve Docker'da yalnızca o aralık açılır."""
    if port == 0 and rtp_ports:
        for p in range(rtp_ports[0], rtp_ports[1] + 1):
            try:
                return RtpSource(host, p, spec.codec, spec.clock_rate, idle_timeout=spec.idle_timeout)
            except OSError:
                continue
        raise OSError(f"RTP port aralığı dolu ({rtp_ports[0]}-{rtp_ports[1]})")
    return RtpSource(host, port, spec.codec, spec.clock_rate, idle_timeout=spec.idle_timeout)


def parse_source(spec: StreamSpec, allow_files: bool, work_dir: Path | None = None,
                 rtp_ports: tuple[int, int] | None = None, max_backlog: float = 30.0):
    """Kaynak yazımından AudioStream nesnesi."""
    if spec.sdp:
        return source_from_sdp(spec.sdp, spec, work_dir)
    k = spec.kaynak.strip()
    if k.lower() == "push":
        return PushSource(spec.ses_hizi, idle_timeout=max(spec.idle_timeout, 5.0), max_backlog=max_backlog)
    u = urlparse(k)
    if u.scheme == "rtp" and not k.lower().endswith(".sdp"):
        if u.port is None:
            raise StreamError("rtp://adres:port biçiminde olmalı (port 0 = boş port seç)")
        if spec.codec not in ("auto", "pcmu", "pcma", "l16"):
            raise StreamError("codec: auto | pcmu | pcma | l16 (Opus vb. için SDP dosyası verin)")
        return bind_rtp(u.hostname or "0.0.0.0", u.port, spec, rtp_ports)  # nosec B104 (RTP dinleme adresi varsayılanı)
    if u.scheme in URL_SCHEMES:
        return FfmpegSource(k)
    p = Path(k)
    if not allow_files:
        raise StreamError("sunucudaki dosya/SDP yolu kapalı (ALLOW_LOCAL_PATHS=1 ile açılır)")
    if not p.is_file():
        raise StreamError(f"dosya yok: {k}")
    if p.suffix.lower() == ".sdp":  # SDP dosyası: biçime göre bizim alıcı ya da ffmpeg
        return source_from_sdp(p.read_text(encoding="utf-8", errors="replace"), spec, work_dir)
    return FfmpegSource(str(p), realtime=True)


@dataclass
class StreamSession:
    id: str
    spec: StreamSpec
    started_at: str
    durum: str = "dinleniyor"          # dinleniyor | bitti | hata | durduruldu
    port: int | None = None
    segments: list[dict] = field(default_factory=list)
    ara: dict | None = None
    ozet: dict | None = None
    hata: str | None = None
    son_is: str | None = None
    kayit: str | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    source: object = None
    subscribers: list[queue_mod.Queue] = field(default_factory=list)
    notifier: object = None
    rtp_adres: str | None = None   # karşı servisin RTP göndereceği adres (STREAM_PUBLIC_HOST ile)

    def to_dict(self, with_segments: bool = True) -> dict:
        d = {"id": self.id, "kaynak": self.spec.kaynak, "dil": self.spec.dil, "durum": self.durum,
             "port": self.port, "baslangic": self.started_at, "parca_sayisi": len(self.segments),
             "ozet": self.ozet, "hata": self.hata, "son_is": self.son_is}
        if self.rtp_adres:
            d["rtp_adres"] = self.rtp_adres
        if isinstance(self.source, PushSource):
            d["ses"] = {"alinan_sn": round(self.source.received, 2), "bekleyen_sn": round(self.source.backlog, 2),
                        "sonraki_sira": self.source.next_seq}
        if with_segments:
            d["parcalar"] = self.segments
            d["ara"] = self.ara
        return d


def segment_dict(s: StreamSegment) -> dict:
    d = {"index": s.index, "start": s.start, "end": s.end, "text": s.text, "final": s.final,
         "confidence": s.conf, "low_confidence_words": list(s.low_conf), "latency": s.latency,
         "kuyruk_bekleme_sn": round(s.kuyruk_bekleme_sn, 3),
         "toplam_gecikme_sn": round(s.toplam_gecikme_sn, 3)}
    if s.duzeltildi:
        d["duzeltildi"] = True
    return d


def port_range(text: str) -> tuple[int, int] | None:
    """"40000-40100" -> (40000, 40100); boş -> None."""
    if not text:
        return None
    a, _, b = text.partition("-")
    lo, hi = int(a), int(b or a)
    if not 1024 <= lo <= hi <= 65535:
        raise ValueError(f"STREAM_RTP_PORTS geçersiz: {text}")
    return lo, hi


class LiveNotifier:
    """Canlı parçaları karşı servisin adresine sırayla POST eder (stream.segment, sonda stream.ended).

    İmza ve başlıklar iş bildirimleriyle aynı (infrastructure/webhook.py). Tek iş parçacığı =
    parçalar sırayla gider. Teslim edilemeyen parça 3 denemeden sonra atılır (canlı akış beklemez;
    kesin metin zaten son tam analizde ve GET /streams/{id}'de durur)."""

    def __init__(self, url: str, secret: str, sid: str, post=None, attempts: int = 3, base_wait: float = 0.5):
        self.url, self.secret, self.sid = url, secret, sid
        self.post, self.attempts, self.base_wait = post or self._post, attempts, base_wait
        self.sent = self.dropped = 0
        self._q: queue_mod.Queue = queue_mod.Queue(maxsize=1000)
        self._t = threading.Thread(target=self._loop, daemon=True, name=f"bildirim-{sid[:8]}")
        self._t.start()

    @staticmethod
    def _post(url: str, body: bytes, headers: dict) -> bool:
        import httpx

        try:
            return httpx.post(url, content=body, headers=headers, timeout=5.0).is_success
        except httpx.HTTPError:
            return False

    def send(self, event: str, payload: dict) -> None:
        try:
            self._q.put_nowait((event, {"stream_id": self.sid, **payload}))
        except queue_mod.Full:
            self.dropped += 1

    def _loop(self) -> None:
        from viet_asr.infrastructure.webhook import build_request

        while True:
            item = self._q.get()
            if item is None:
                return
            event, payload = item
            for i in range(self.attempts):
                body, headers = build_request(payload, event, self.sid, uuid.uuid4().hex, self.secret)
                if self.post(self.url, body, headers):
                    self.sent += 1
                    break
                time.sleep(self.base_wait * 2 ** i)
            else:
                self.dropped += 1

    def close(self, wait: float = 15.0) -> None:
        self._q.put(None)
        self._t.join(wait)


class StreamManager:
    def __init__(self, engines, vad_cfg: VadConfig, data_dir: str | Path, submit=None,
                 languages: list[str] | None = None, max_sessions: int = 8, allow_files: bool = False,
                 max_speech: float = 10.0, vad_factory=None, rtp_ports: tuple[int, int] | None = None,
                 public_host: str = "", notify_secret: str = "", notify_post=None, max_backlog: float = 30.0,
                 fast_engines=None, stream_two_stage: bool = False, two_stage_max: int = 1):
        self.engines, self.vad_cfg, self.submit = engines, vad_cfg, submit
        self.fast_engines = fast_engines
        self.stream_two_stage = stream_two_stage
        # Aynı anda Whisper düzeltmesi alan akış sayısı üst sınırı (0 = sınırsız). Medium düzeltmesi CPU'da
        # yalnızca tek akışta güvenilir (docs/olcumler.md); diğer akışlar yalnız hızlı modelle devam eder.
        self.two_stage_max = two_stage_max
        self._two_stage_active = 0
        self.dir = Path(data_dir) / "akislar"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.languages = languages or list(engines)
        self.max_sessions, self.allow_files, self.max_speech = max_sessions, allow_files, max_speech
        self.rtp_ports, self.public_host, self.max_backlog = rtp_ports, public_host, max_backlog
        self.notify_secret, self.notify_post = notify_secret, notify_post
        self.vad_factory = vad_factory or (lambda max_speech=None: SileroStreamingVad(
            self.vad_cfg, max_speech=max_speech or self.max_speech))
        self.sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, s, engines, vad_cfg, submit=None, fast_engines=None) -> "StreamManager":
        """Tek yer: yerel mod ve tam mod aynı ayarlarla kurar."""
        two_stage = getattr(s, "stream_two_stage", False)
        if fast_engines is None and two_stage:
            from viet_asr.infrastructure.catalog import language_profile
            from viet_asr.infrastructure.engines import LazyAsrEngines
            from viet_asr.infrastructure.normalizer import normalizer_for
            from viet_asr.infrastructure.sherpa_asr import build_asr

            threads = getattr(s, "threads", 2)
            fast_model = getattr(s, "stream_fast_model", "seda-v0.1")
            models_dir = getattr(s, "models_dir", "models")
            fast_engines = LazyAsrEngines({
                "tr": lambda: build_asr(
                    language_profile("tr", models_dir, fast_model, num_threads=threads),
                    normalizer_for("tr"))
            })
        return cls(engines, vad_cfg, s.data_dir, submit=submit, languages=list(s.languages),
                   max_sessions=s.stream_max_sessions, allow_files=s.allow_local_paths,
                   max_speech=s.stream_max_speech, rtp_ports=port_range(s.stream_rtp_ports),
                   public_host=s.stream_public_host, notify_secret=s.webhook_secret,
                   max_backlog=s.stream_max_backlog, fast_engines=fast_engines,
                   stream_two_stage=two_stage, two_stage_max=getattr(s, "stream_two_stage_max", 1))

    def active(self) -> int:
        return sum(1 for s in self.sessions.values() if s.durum == "dinleniyor")

    def start(self, spec: StreamSpec) -> StreamSession:
        if spec.dil not in self.languages:
            raise StreamError(f"dil '{spec.dil}' yok; seçenekler: {', '.join(self.languages)}")
        with self._lock:
            if self.active() >= self.max_sessions:
                raise StreamError(f"en fazla {self.max_sessions} canlı akış (STREAM_MAX_SESSIONS)")
            source = parse_source(spec, self.allow_files, self.dir, self.rtp_ports, self.max_backlog)
            self._prune()
            sid = uuid.uuid4().hex
            s = StreamSession(sid, spec, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              port=getattr(source, "port", None), source=source)
            if s.port and self.public_host:
                s.rtp_adres = f"{self.public_host}:{s.port}"
            if spec.canli_url:
                s.notifier = LiveNotifier(spec.canli_url, self.notify_secret, sid, self.notify_post)
            self.sessions[sid] = s
        threading.Thread(target=self._run, args=(s,), daemon=True, name=f"akis-{sid[:8]}").start()
        return s

    def _prune(self, keep: int = 100) -> None:
        """Biten oturumların en eskilerini bellekten atar (kayıt ve tam analiz işi kalır)."""
        done = [k for k, v in self.sessions.items() if v.durum != "dinleniyor"]
        for k in done[:max(0, len(done) - keep)]:
            del self.sessions[k]

    def get(self, sid: str) -> StreamSession | None:
        return self.sessions.get(sid)

    def stop(self, sid: str) -> StreamSession | None:
        s = self.sessions.get(sid)
        if s is not None and s.durum == "dinleniyor":
            s.stop.set()
            s.source.close()
        return s

    def push(self, sid: str, data: bytes, seq: int | None = None) -> StreamSession:
        """Push kaynağına ses parçası (HTTP/WS). Hatalar: KeyError (yok), StreamError (push değil/kapalı),
        BacklogFull (yavaşla), SeqError (atlanan parça)."""
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        if not isinstance(s.source, PushSource):
            raise StreamError("bu akış push kaynağı değil (kaynak: \"push\")")
        if s.durum != "dinleniyor":
            raise StreamError(f"akış {s.durum}")
        s.source.feed(data, seq)
        return s

    def end_push(self, sid: str) -> None:
        s = self.sessions.get(sid)
        if s is not None and isinstance(s.source, PushSource):
            s.source.end()

    def subscribe(self, sid: str, after: int = -1) -> queue_mod.Queue | None:
        """after: bu sıradaki ve öncesindeki kesin parçaları yeniden gönderme (bağlantı kopup dönenler)."""
        s = self.sessions.get(sid)
        if s is None:
            return None
        q: queue_mod.Queue = queue_mod.Queue()
        for seg in s.segments:  # geç bağlanan da baştan görsün
            if seg["index"] > after:
                tur = "duzeltme" if seg.get("duzeltildi") else "parca"
                q.put({"tur": tur, **seg})
        if s.durum != "dinleniyor":
            q.put({"tur": "bitti", **s.to_dict(with_segments=False)})
        s.subscribers.append(q)
        return q

    def _publish(self, s: StreamSession, event: dict) -> None:
        for q in list(s.subscribers):
            q.put(event)
        n = s.notifier
        if n is not None:
            if event["tur"] == "parca" and event.get("final"):
                n.send("stream.segment", {k: v for k, v in event.items() if k != "tur"})
            elif event["tur"] == "duzeltme":
                n.send("stream.correction", {k: v for k, v in event.items() if k != "tur"})
            elif event["tur"] == "bitti":
                n.send("stream.ended", {k: v for k, v in event.items() if k != "tur"})
                threading.Thread(target=n.close, daemon=True).start()

    def _run(self, s: StreamSession) -> None:
        path = self.dir / f"{s.id}.wav"
        wav = sf.SoundFile(str(path), "w", samplerate=16000, channels=1, subtype="PCM_16")
        s.kayit = str(path)

        def on_segment(seg: StreamSegment) -> None:
            d = segment_dict(seg)
            if seg.duzeltildi:
                for i, ex in enumerate(s.segments):
                    if ex["index"] == seg.index:
                        s.segments[i] = d
                        break
                else:
                    s.segments.append(d)
                self._publish(s, {"tur": "duzeltme", **d})
            elif seg.final:
                s.segments.append(d)
                s.ara = None
                self._publish(s, {"tur": "parca", **d})
            else:
                s.ara = d
                self._publish(s, {"tur": "parca", **d})

        slot = False  # iki aşamalı düzeltme kotasından yer aldı mı
        kota_doldu = False
        try:
            two_stage = (s.spec.iki_asama if s.spec.iki_asama is not None else self.stream_two_stage) and \
                        self.fast_engines is not None and s.spec.dil in self.fast_engines
            fast_asr = None
            if two_stage:
                try:  # hızlı model yüklenemezse (indirilmemiş, bozuk) iki aşamasız, eski davranışla devam et
                    fast_asr = self.fast_engines[s.spec.dil]
                except Exception as e:
                    log.warning("akış %s: hızlı model yüklenemedi (%s: %s), iki aşamasız devam", s.id,
                                type(e).__name__, e)
                    two_stage = False
            if two_stage:
                asr = fast_asr
                if s.spec.duzeltme:
                    with self._lock:
                        if self.two_stage_max <= 0 or self._two_stage_active < self.two_stage_max:
                            self._two_stage_active += 1
                            slot = True
                if not s.spec.duzeltme:
                    refiner = None  # istek üzerine: yalnız hızlı model
                elif slot:
                    refiner = self.engines[s.spec.dil]
                else:  # kota dolu: Whisper düzeltmesi yok, yalnız hızlı model (CPU'yu boğmasın)
                    refiner, kota_doldu = None, True
                    log.info("akış %s: iki aşamalı kota dolu (%d), yalnız hızlı modelle devam", s.id, self.two_stage_max)
            else:
                asr = self.engines[s.spec.dil]
                refiner = None
            fast = any(k in getattr(asr, "name", "") for k in FAST_ENGINES)
            st = StreamTranscriber(self.vad_factory)
            t0 = time.monotonic()
            sm = st.run(s.source, asr, on_segment, record=wav.write, interim=fast, stop=s.stop,
                        refiner=refiner)
            s.ozet = {"sure_sn": round(sm.duration, 1), "parca": sm.segments, "kelime": sm.words,
                      "gecikme_p95_sn": round(sm.p95_latency, 2), "gecikme_max_sn": round(sm.max_latency, 2),
                      "kuyruk_bekleme_max_sn": round(sm.max_kuyruk_bekleme, 2),
                      "toplam_gecikme_p95_sn": round(sm.p95_toplam_gecikme, 2),
                      "toplam_gecikme_max_sn": round(sm.max_toplam_gecikme, 2),
                      "duzeltmeler": sm.duzeltmeler, "duzeltme_pencere": sm.duzeltme_pencere,
                      "iki_asama": refiner is not None, "iki_asama_kota_doldu": kota_doldu,
                      "duvar_sn": round(time.monotonic() - t0, 1),
                      **({("push" if isinstance(s.source, PushSource) else "rtp"): dict(s.source.stats)}
                         if hasattr(s.source, "stats") else {})}
            s.durum = "durduruldu" if s.stop.is_set() else "bitti"
        except Exception as e:  # kaynak açılamadı, ffmpeg hatası, model yok...
            log.exception("akış %s hata", s.id)
            s.durum, s.hata = "hata", f"{type(e).__name__}: {e}"
        finally:
            wav.close()
            if slot:
                with self._lock:
                    self._two_stage_active -= 1
        dur = (s.ozet or {}).get("sure_sn", 0)
        if s.spec.son_analiz and self.submit is not None and dur >= 1.0:
            try:
                s.son_is = self.submit(AudioJob(uuid.uuid4().hex, str(path), s.spec.dil, s.spec.callback_url,
                                                s.spec.num_speakers, external_id=f"akis:{s.id}"))
            except Exception as e:
                log.warning("akış %s tam analize verilemedi: %s", s.id, e)
        elif dur < 1.0:
            path.unlink(missing_ok=True)
            s.kayit = None
        log.info("akış %s %s: %s", s.id, s.durum, s.ozet)
        self._publish(s, {"tur": "bitti", **s.to_dict(with_segments=False)})
