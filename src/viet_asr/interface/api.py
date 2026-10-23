"""HTTP API (FastAPI).

    POST /jobs          ses dosyası yükle -> 202 + job_id (iş kuyruğa girer)
                        form alanları: file (zorunlu), language (vi | tr | auto, varsayılan auto),
                        callback_url (isteğe bağlı: iş bitince buraya POST edilir),
                        num_speakers (isteğe bağlı: biliniyorsa konuşmacı sayısı; yoksa sesten bulunur)
                        external_id (isteğe bağlı: çağıran sistemin kendi kimliği; aynısı tekrar
                        gelirse yeni iş açılmaz, var olan iş 200 ile döner -> ağ hatasında
                        güvenle tekrar gönderilebilir)
                        profile (full | feedback; feedback: kısa mesajlar için hızlı)
                        notify (callback olayları, virgülle: running,done,failed,rejected)
    GET  /jobs          son işler (?limit=20; arayüzdeki geçmiş)
    GET  /jobs/{id}     durum; bittiyse sonuç (konuşmacılı satırlar)
    POST /streams {"kaynak": "rtp://0.0.0.0:5004", "dil": "vi"}  canlı akış (RTP, URL, SDP)
                        ya da {"sdp": "v=0 ... m=audio 5004 RTP/AVP 96 ... a=rtpmap:96 L16/16000"}
    GET  /streams/{id}   canlı parçalar; WS /streams/{id}/ws anlık; DELETE /streams/{id} durdurur
    GET  /jobs/{id}/transcript  temiz transkript (?etiket=genel|studyo|yok&temizle=1&noktalama=1&bicim=txt|json)
    GET  /jobs/{id}/audio   yüklenen ses (saklama süresi dolmadıysa; arayüzde dinlemek için)
    DELETE /jobs/{id}   bitmiş işi, sonucunu ve sesini siler (kuyruktaki/işlenen iş silinmez)
    POST /jobs/{id}/dataset {"reviewer","consent","use","notes"}  veri setinin onay kuyruğuna
    POST /jobs/{id}/reanalyze {"num_speakers"}  aynı sesi yeni konuşmacı sayısıyla yeniden analiz et
    DELETE /jobs/{id}/utterances/{n}  n. satırı sonuçtan kaldır (uydurma satır; kayıtta kalır)
    PATCH /jobs/{id}/utterances/{n}   n. satırın metnini/konuşmacısını düzelt ({"text", "speaker"});
                        modelin ilk hali saklanır, düzeltilen işin sesi silinmez (ince ayar verisi)
    POST /webhooks/test {"url"}   imzalı deneme bildirimi (entegrasyon kurulurken)
    GET  /webhooks/dead  teslim edilemeyen bildirimler; POST /webhooks/{id}/retry yeniden dener
    GET  /corrections   düzeltilen satırlar (scripts/duzeltmeleri_disa_aktar.py ince ayar verisine çevirir)
    GET  /              web arayüzü (dosya yükle, dil seç, sonucu dinle/indir)
    GET  /health        veritabanı ve kuyruk erişimi, durum sayıları (anahtar istemez)
    GET  /health/ready  hazır mı: modeller yüklü / her kuyrukta çalışan işçi var (anahtar istemez)
    GET  /metrics       gecikme ve "1 dk ses <= 1 dk" hedefi, kuyruk doluluğu (JSON)
    GET  /metrics/prometheus   aynı ve daha fazlası Prometheus biçiminde (aşama süreleri, webhook)

API_KEY tanımlıysa /jobs istekleri "X-API-Key" başlığı ister (production'da zorunlu).

Çalıştırma:  uvicorn viet_asr.interface.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from viet_asr.application.jobs import SubmitJob
from viet_asr.application.maintenance import latency_stats
from viet_asr.application.ports import JobQueue, JobStore
from viet_asr.application.serialization import record_to_dict
from viet_asr.domain import AudioJob, JobStatus, UnsupportedLanguageError
from viet_asr.infrastructure.audio import FFMPEG_ONLY, AudioDecodeError, ffmpeg_bin, probe_duration
from viet_asr.interface.composition import audio_roots, safe_file_remover
from viet_asr.interface.security import api_key_dependency, callback_problem, key_ok, stream_source_problem
from viet_asr.interface.settings import Settings

log = logging.getLogger(__name__)
_ALLOWED = {".wav", ".flac", ".ogg", ".mp3", ".opus"} | FFMPEG_ONLY
_UI = Path(__file__).parent / "static" / "index.html"


class DatasetExport(BaseModel):
    reviewer: str = Field(..., min_length=2, max_length=100)   # kontrol eden kişi
    consent: bool                                              # kaydın kullanımına onay
    use: str = Field("test", pattern="^(test|egitim)$")        # test: ölçüm, egitim: ince ayar
    notes: str | None = Field(None, max_length=1000)


class StreamStart(BaseModel):
    kaynak: str = Field("", max_length=500)                   # rtp://0.0.0.0:5004, URL, .sdp
    # SDP metni (santralin verdiği): port ve biçim buradan; PCMU/PCMA/L16 bizim alıcı, Opus/G.722 ffmpeg
    sdp: str | None = Field(None, max_length=20000)
    dil: str = Field("vi", max_length=8)
    codec: str = Field("auto", pattern="^(auto|pcmu|pcma|l16)$")
    clock_rate: int | None = Field(None, ge=8000, le=48000)
    iki_asama: bool | None = None                             # None: sunucu ayarı (STREAM_TWO_STAGE); False: tek model (Whisper)
    duzeltme: bool = True                                     # False: iki aşamalıda Whisper düzeltmesini kapat (yalnız hızlı model)
    son_analiz: bool = True                                   # bitince tam analiz (konuşmacı ayırma)
    num_speakers: int | None = Field(None, ge=1, le=20)
    callback_url: str | None = None
    idle_timeout: float = Field(5.0, ge=1.0, le=300.0)        # bu kadar paket gelmezse akış biter
    ses_hizi: int = Field(16000, ge=8000, le=48000)           # kaynak "push": gönderilen PCM16 hızı
    canli_url: str | None = None                              # canlı parçaların POST edileceği adres (imzalı)


class Reanalyze(BaseModel):
    num_speakers: int = Field(..., ge=0, le=20)


class WebhookTest(BaseModel):
    url: str = Field(..., max_length=2000)


class UtteranceEdit(BaseModel):
    text: str | None = Field(None, max_length=5000)
    # konuşmacı etiketi: speaker_00, BELIRSIZ ya da arayüzde verilen kısa bir ad (ör. doktor)
    speaker: str | None = Field(None, pattern=r"^[\w\- ]{1,40}$")


def create_app(store: JobStore, queue: JobQueue, settings: Settings,
               health_checks: dict | None = None, queue_stats=None, rate_limiter=None,
               readiness: dict | None = None, streams=None) -> FastAPI:
    """readiness: ad -> () -> (hazır mı, açıklama); /health/ready bunları da sorar."""
    api = FastAPI(title="Ses -> metin + konuşmacı (vi, tr)", version="0.3.0")
    auth = Depends(api_key_dependency(settings))
    submit = SubmitJob(store, queue, settings.languages)
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    remove_upload = safe_file_remover(data_dir)

    _recent: dict[tuple, float] = {}

    def log_access(who: str, action: str, job_id: str | None, request: Request,
                   detail: str | None = None, dedupe: bool = False) -> None:
        """Erişim kaydı. dedupe: aynı kişinin aynı işteki aynı eylemi 60 sn'de bir yazılır
        (arayüz çalışan işi saniyede bir sorar, ses oynatıcı parça parça ister)."""
        import time as _t

        if dedupe:
            key, now = (who, action, job_id), _t.monotonic()
            if now - _recent.get(key, -1e9) < 60:
                return
            if len(_recent) > 10_000:
                _recent.clear()
            _recent[key] = now
        try:
            store.audit(who, action, job_id, detail, request.client.host if request.client else None)
        except Exception:  # kayıt yazılamadı: istek yine de yanıtlansın, hata günlükte
            log.exception("erişim kaydı yazılamadı: %s %s %s", who, action, job_id)

    if rate_limiter is None and settings.rate_limit_per_minute > 0:
        from viet_asr.infrastructure.rate_limit import MemoryRateLimiter

        rate_limiter = MemoryRateLimiter(settings.rate_limit_per_minute)

    def admit(request: Request):
        """Yükleme işlenmeden önce: istek sınırı ve bekleyen iş üst sınırı."""
        if rate_limiter is not None:
            key = request.headers.get("X-API-Key") or (request.client.host if request.client else "?")
            wait = rate_limiter.check(hashlib.sha256(key.encode()).hexdigest()[:16])
            if wait is not None:
                raise HTTPException(429, f"dakikada en fazla {settings.rate_limit_per_minute} istek",
                                    headers={"Retry-After": str(wait)})
        if settings.max_pending_jobs > 0:
            c = store.counts()
            pending = c.get(JobStatus.QUEUED.value, 0) + c.get(JobStatus.RUNNING.value, 0)
            if pending >= settings.max_pending_jobs:
                raise HTTPException(503, f"sistem dolu: {pending} iş bekliyor; biraz sonra tekrar deneyin",
                                    headers={"Retry-After": "60"})

    @api.post("/jobs", status_code=202, dependencies=[Depends(admit)])
    def create_job(request: Request, who: str = auth, file: UploadFile | None = File(None),
                   language: str = Form("auto"),
                   callback_url: str | None = Form(None), audio_path: str | None = Form(None),
                   num_speakers: int | None = Form(None), external_id: str | None = Form(None),
                   profile: str | None = Form(None), notify: str | None = Form(None)):
        problem = callback_problem(callback_url, settings)
        if problem:
            raise HTTPException(422, problem)
        job_id = uuid.uuid4().hex
        if file is not None:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in _ALLOWED:
                raise HTTPException(415, f"desteklenmeyen dosya türü '{ext}'; izinli: {sorted(_ALLOWED)}")
            if ext in FFMPEG_ONLY and ffmpeg_bin() is None:
                raise HTTPException(415, f"'{ext}' için sunucuda ffmpeg kurulu olmalı")
            path = data_dir / f"{job_id}{ext}"
            written, too_big = 0, False
            with path.open("wb") as out:
                while chunk := file.file.read(1 << 20):
                    written += len(chunk)
                    if written > max_bytes:  # sınır aşılınca dur: devasa yükleme diski doldurmasın
                        too_big = True
                        break
                    out.write(chunk)
            if too_big:
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"dosya {settings.max_upload_mb} MB'tan büyük")
        elif audio_path and settings.allow_local_paths:
            path = Path(audio_path)
            if not path.is_file():
                raise HTTPException(404, f"dosya yok: {audio_path}")
        else:
            raise HTTPException(422, "file alanı zorunlu")
        def reject(code, msg):
            if file is not None:
                path.unlink(missing_ok=True)
            raise HTTPException(code, msg)

        try:
            duration = probe_duration(str(path))  # bozuk dosya kuyruğa girmesin
        except (AudioDecodeError, Exception):
            reject(400, "ses dosyası okunamadı (bozuk ya da desteklenmeyen biçim)")
        if duration > settings.max_audio_minutes * 60:
            reject(413, f"ses {settings.max_audio_minutes:g} dakikadan uzun")
        try:
            got = submit(AudioJob(job_id, str(path.resolve()), language, callback_url, num_speakers,
                                  external_id, profile,
                                  tuple(notify.split(",")) if notify is not None else None))
        except (UnsupportedLanguageError, ValueError) as e:
            reject(422, str(e))
        if got != job_id:  # aynı external_id daha önce gönderilmiş: yeni dosya gereksiz
            if file is not None:
                path.unlink(missing_ok=True)
            rec = store.get(got)
            return JSONResponse({"job_id": got, "status": rec.status.value if rec else None,
                                 "tekrar": True}, status_code=200)
        log_access(who, "is_olusturuldu", job_id, request, f"{duration:.0f} sn, dil={language}")
        return {"job_id": job_id, "status": JobStatus.QUEUED.value,
                "duration": round(duration, 3), "language": language}

    @api.get("/", include_in_schema=False)
    def ui():
        return FileResponse(_UI, media_type="text/html; charset=utf-8")

    @api.get("/jobs", dependencies=[auth])
    def list_jobs(limit: int = 20):
        return [record_to_dict(r) for r in store.recent_jobs(max(1, min(limit, 200)))]

    # --- canlı akış ---
    def _streams():
        if streams is None or not settings.enable_streams:
            raise HTTPException(404, "canlı akış bu modda kapalı (ENABLE_STREAMS=1, yerel mod ya da akış CLI)")
        return streams

    @api.post("/streams", status_code=201, dependencies=[auth])
    def start_stream(body: StreamStart):
        """Canlı akışı dinlemeye başlar; metin GET /streams/{id} ve WS /streams/{id}/ws ile gelir."""
        from viet_asr.interface.streams import StreamError, StreamSpec

        problem = (callback_problem(body.callback_url, settings) or callback_problem(body.canli_url, settings)
                   or stream_source_problem(body.kaynak, settings))
        if problem:
            raise HTTPException(422, problem)
        if not body.sdp and len(body.kaynak.strip()) < 3:
            raise HTTPException(422, "kaynak ya da sdp verilmeli")
        try:
            s = _streams().start(StreamSpec(**body.model_dump()))
        except StreamError as e:
            raise HTTPException(422, str(e))
        except OSError as e:  # port kullanımda vb.
            raise HTTPException(409, f"kaynak açılamadı: {e}")
        return s.to_dict()

    @api.get("/streams", dependencies=[auth])
    def list_streams():
        return [s.to_dict(with_segments=False) for s in _streams().sessions.values()]

    @api.get("/streams/{sid}", dependencies=[auth])
    def get_stream(sid: str):
        s = _streams().get(sid)
        if s is None:
            raise HTTPException(404, "akış bulunamadı")
        return s.to_dict()

    @api.delete("/streams/{sid}", dependencies=[auth])
    def stop_stream(sid: str):
        s = _streams().stop(sid)
        if s is None:
            raise HTTPException(404, "akış bulunamadı")
        return {"id": sid, "durum": "durduruluyor" if s.durum == "dinleniyor" else s.durum}

    @api.post("/streams/{sid}/audio", status_code=202, dependencies=[auth])
    async def push_audio(sid: str, request: Request, seq: int | None = None, son: bool = False):
        """Parça parça ses (kaynak "push"): gövde ham PCM16 little-endian, tek kanal, hız = akışın ses_hizi.
        seq: 0'dan artan sıra no (tekrar gönderilen parça atlanır). son=true: akış biter.
        429 + Retry-After: işleme yetişilemiyor, bekleyip aynı parçayı tekrar gönderin."""
        from viet_asr.infrastructure.push_source import BacklogFull, SeqError
        from viet_asr.interface.streams import StreamError

        mgr = _streams()
        data = await request.body()
        if len(data) > 2_000_000:
            raise HTTPException(413, "parça en fazla 2 MB (yaklaşık 60 sn @16 kHz)")
        try:
            s = mgr.push(sid, data, seq)
        except KeyError:
            raise HTTPException(404, "akış bulunamadı")
        except StreamError as e:
            raise HTTPException(409, str(e))
        except BacklogFull as e:
            raise HTTPException(429, str(e), headers={"Retry-After": "1"})
        except SeqError as e:
            raise HTTPException(409, f"{e}; araya giren parça var", headers={"X-Expected-Seq": str(e.expected)})
        except ValueError as e:
            raise HTTPException(409, str(e))
        if son:
            mgr.end_push(sid)
        src = s.source
        return {"alinan_sn": round(src.received, 2), "bekleyen_sn": round(src.backlog, 2),
                "sonraki_sira": src.next_seq}

    @api.get("/streams/{sid}/events", dependencies=[auth])
    async def stream_events(sid: str, request: Request, after: int = -1):
        """Sunucu-gönderimli olaylar (SSE): her kesin parça {"tur":"parca",...}, sonda "bitti".
        Kesin parçanın id'si = sırası; kopup dönen istemci Last-Event-ID (ya da ?after=) ile kaldığı yerden alır."""
        import asyncio
        import json as json_mod
        import queue as queue_mod

        from fastapi.responses import StreamingResponse

        mgr = _streams()
        last = request.headers.get("last-event-id")
        if last is not None and last.lstrip("-").isdigit():
            after = max(after, int(last))
        q = mgr.subscribe(sid, after)
        if q is None:
            raise HTTPException(404, "akış bulunamadı")

        async def gen():
            idle = 0.0
            try:
                while True:
                    try:
                        ev = q.get_nowait()
                    except queue_mod.Empty:
                        if await request.is_disconnected():
                            return
                        await asyncio.sleep(0.05)
                        idle += 0.05
                        if idle >= 15:
                            idle = 0.0
                            yield ": canli\n\n"
                        continue
                    idle = 0.0
                    sid_line = f"id: {ev['index']}\n" if ev.get("tur") in ("parca", "duzeltme") and ev.get("final") else ""
                    yield f"{sid_line}event: {ev['tur']}\ndata: {json_mod.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev.get("tur") == "bitti":
                        return
            finally:
                s = mgr.get(sid)
                if s is not None and q in s.subscribers:
                    s.subscribers.remove(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.websocket("/streams/{sid}/audio")
    async def stream_audio_ws(ws: WebSocket, sid: str, key: str | None = None):
        """Tek bağlantıda ses yukarı, metin aşağı: ikili mesaj = ham PCM16 parçası (akışın ses_hizi),
        "son" metni ya da bağlantıyı kapatmak akışı bitirir. Sunucu aynı bağlantıdan JSON olayları yollar."""
        import asyncio
        import queue as queue_mod

        from viet_asr.infrastructure.push_source import BacklogFull
        from viet_asr.interface.streams import StreamError

        if not key_ok(settings, key or ws.headers.get("x-api-key")):
            await ws.close(code=4401)
            return
        mgr = streams if (streams is not None and settings.enable_streams) else None
        q = mgr.subscribe(sid) if mgr else None
        if q is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        done = asyncio.Event()

        async def down():
            while not done.is_set():
                try:
                    ev = q.get_nowait()
                except queue_mod.Empty:
                    await asyncio.sleep(0.05)
                    continue
                await ws.send_json(ev)
                if ev.get("tur") == "bitti":
                    done.set()

        sender = asyncio.create_task(down())
        try:
            while not done.is_set():
                recv = asyncio.create_task(ws.receive())
                await asyncio.wait({recv, asyncio.create_task(done.wait())}, return_when=asyncio.FIRST_COMPLETED)
                if not recv.done():
                    recv.cancel()
                    break
                msg = recv.result()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    for _ in range(100):  # geri basınç: kuyruk boşalana kadar (en çok ~10 sn) bekle
                        try:
                            mgr.push(sid, msg["bytes"])
                            break
                        except BacklogFull:
                            await asyncio.sleep(0.1)
                        except (KeyError, StreamError, ValueError):
                            done.set()
                            break
                elif (msg.get("text") or "").strip().lower() in ("son", "end"):
                    break
        except WebSocketDisconnect:
            pass
        finally:
            mgr.end_push(sid)
            if not done.is_set():  # son sonuçlar gelene kadar en çok 30 sn bekle
                try:
                    await asyncio.wait_for(done.wait(), 30)
                except asyncio.TimeoutError:
                    pass
            sender.cancel()
            s = mgr.get(sid)
            if s is not None and q in s.subscribers:
                s.subscribers.remove(q)
            try:
                await ws.close()
            except RuntimeError:
                pass

    @api.websocket("/streams/{sid}/ws")
    async def stream_ws(ws: WebSocket, sid: str, key: str | None = None):
        """Canlı olaylar: {"tur": "parca", final, text, start, end...} ve sonda {"tur": "bitti"}.
        API anahtarı tarayıcıdan başlık gönderilemediği için ?key= ile."""
        import asyncio
        import queue as queue_mod

        if not key_ok(settings, key or ws.headers.get("x-api-key")):
            await ws.close(code=4401)
            return
        mgr = streams if (streams is not None and settings.enable_streams) else None
        q = mgr.subscribe(sid) if mgr else None
        if q is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        try:
            while True:
                try:
                    ev = q.get_nowait()
                except queue_mod.Empty:
                    await asyncio.sleep(0.05)
                    continue
                await ws.send_json(ev)
                if ev.get("tur") == "bitti":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            s = mgr.get(sid)
            if s is not None and q in s.subscribers:
                s.subscribers.remove(q)
        try:
            await ws.close()
        except RuntimeError:
            pass

    @api.get("/jobs/{job_id}/transcript")
    def get_transcript_view(job_id: str, request: Request, who: str = auth, etiket: str | None = None,
                            temizle: bool = True, noktalama: bool = True, bicim: str = "txt"):
        """Temiz transkript: konuşmacı değişiminde yeni satır, [dd:ss - dd:ss] [Etiket]: metin.
        etiket: genel | studyo | yok (varsayılan TRANSCRIPT_LABELS). Kayıttaki satırlar değişmez."""
        from viet_asr.application.transcript_view import LABEL_MODES, as_text, blocks

        mode = etiket or settings.transcript_labels
        if mode not in LABEL_MODES or bicim not in ("txt", "json"):
            raise HTTPException(422, f"etiket {LABEL_MODES} biri, bicim txt ya da json olmalı")
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(404, "iş bulunamadı")
        t = store.get_transcript(job_id) if rec.status == JobStatus.DONE else None
        if t is None:
            raise HTTPException(409, "iş henüz bitmedi")
        bs = blocks(t, mode, clean=temizle, punctuate=noktalama)
        log_access(who, "transkript_alindi", job_id, request, dedupe=True)
        if bicim == "json":
            return [{"speaker": b.speaker, "label": b.label, "start": round(b.start, 2),
                     "end": round(b.end, 2), "text": b.text} for b in bs]
        return PlainTextResponse(as_text(bs), media_type="text/plain; charset=utf-8")

    @api.get("/jobs/{job_id}/audio")
    def get_audio(job_id: str, request: Request, who: str = auth):
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(404, "iş bulunamadı")
        p = Path(rec.job.audio_path) if rec.job.audio_path else None
        if p is None or not p.is_file():
            raise HTTPException(410, "ses dosyası saklama süresi dolduğu için silinmiş")
        roots = audio_roots(settings)  # yüklemeler + canlı oturum kayıtları
        if not any(r in p.resolve().parents for r in roots) and not settings.allow_local_paths:
            raise HTTPException(403, "bu ses dosyası sunulamaz")
        log_access(who, "ses_dinlendi", job_id, request, dedupe=True)
        return FileResponse(p)

    @api.delete("/jobs/{job_id}")
    def delete_job(job_id: str, request: Request, who: str = auth):
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(404, "iş bulunamadı")
        if rec.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise HTTPException(409, "iş henüz bitmedi; bitince silinebilir")
        path = store.delete_job(job_id)
        log_access(who, "is_silindi", job_id, request)
        if path:
            remove_upload(path)  # yalnızca yükleme klasöründekiler; kullanıcının kendi dosyası kalır
        return {"job_id": job_id, "silindi": True}

    @api.patch("/jobs/{job_id}/utterances/{idx}")
    def edit_utterance(job_id: str, idx: int, body: UtteranceEdit, request: Request, who: str = auth):
        if body.speaker is None and body.text is None:
            raise HTTPException(422, "text ya da speaker verilmeli")
        if body.text is not None and not body.text.strip():
            raise HTTPException(422, "metin boş olamaz; yanlış satırı kaldırmak için satırı silin")
        u = store.update_utterance(job_id, idx, body.speaker,
                                   " ".join(body.text.split()) if body.text is not None else None)
        if u is None:
            raise HTTPException(404, "bitmiş iş ya da satır bulunamadı")
        log_access(who, "satir_duzeltildi", job_id, request,
                   f"satır {idx}" + (" metin" if body.text is not None else "") + (" konuşmacı" if body.speaker else ""))
        return {"index": idx, "speaker": u.speaker, "start": round(u.start, 3), "end": round(u.end, 3),
                "text": u.text, "edited": True}

    @api.post("/jobs/{job_id}/reanalyze", status_code=202)
    def reanalyze(job_id: str, body: Reanalyze, request: Request, who: str = auth):
        """Aynı ses, yeni konuşmacı sayısıyla yeniden analiz (dosya tekrar yüklenmez; sunucudaki
        ses kullanılır). Aynı iş kimliği kalır; elle yapılan düzeltmeler yeni sonuçla değişir.
        num_speakers=0: sesten bul."""
        rec = store.reopen(job_id, body.num_speakers or None)
        if rec is None:
            raise HTTPException(409, "yalnızca bitmiş ve sesi silinmemiş iş yeniden analiz edilebilir")
        log_access(who, "yeniden_analiz", job_id, request, f"konuşmacı sayısı {body.num_speakers or 'otomatik'}")
        code = rec.decision.code if rec.decision else rec.job.language
        if code:
            queue.enqueue_transcription(job_id, code)
        else:
            queue.enqueue_detection(job_id)
        return {"job_id": job_id, "status": JobStatus.QUEUED.value, "num_speakers": rec.job.num_speakers}

    @api.post("/jobs/{job_id}/dataset")
    def to_dataset(job_id: str, body: DatasetExport, request: Request, who: str = auth):
        """Kontrol edilmiş sonucu veri setinin onay kuyruğuna koyar (scripts/veri_seti.py onaylar).
        ENABLE_DATASET_EXPORT açık olmalı; kontrol eden kişinin adı ve açık onay zorunlu."""
        if not settings.enable_dataset_export:
            raise HTTPException(403, "veri setine aktarma kapalı (ENABLE_DATASET_EXPORT)")
        if not body.consent:
            raise HTTPException(422, "kaydın veri setinde kullanılmasına onay verilmeli")
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(404, "iş bulunamadı")
        t = store.get_transcript(job_id) if rec.status == JobStatus.DONE else None
        if t is None:
            raise HTTPException(409, "yalnızca bitmiş iş aktarılabilir")
        p = Path(rec.job.audio_path) if rec.job.audio_path else None
        if p is None or not p.is_file():
            raise HTTPException(410, "ses dosyası silinmiş (saklama süresi)")
        from viet_asr.application.serialization import transcript_to_dict
        from viet_asr.infrastructure.audio import SoundfileLoader
        from viet_asr.infrastructure.dataset import write_pending

        res = transcript_to_dict(t)
        meta = {"kontrol_eden": body.reviewer.strip(), "kullanim": body.use, "not": body.notes or "",
                "dil": t.language.code if t.language else None, "model": t.model,
                "duzeltilen_satir": sum(1 for u in t.utterances if u.edited)}
        d = write_pending(Path(settings.dataset_dir), job_id, SoundfileLoader().load(str(p)),
                          res["utterances"], meta)
        store.set_consent(job_id, body.reviewer.strip(), body.use)
        log_access(who, "veri_setine_aktarildi", job_id, request, f"{body.use}, kontrol eden: {body.reviewer.strip()}")
        return {"job_id": job_id, "durum": "onay bekliyor", "klasor": str(d)}

    @api.delete("/jobs/{job_id}/utterances/{idx}")
    def delete_utterance(job_id: str, idx: int, request: Request, who: str = auth):
        """Uydurma/yanlış satırı sonuçtan kaldırır; kayıtta kalır (model hatası olarak incelenir)."""
        if not store.delete_utterance(job_id, idx):
            raise HTTPException(404, "bitmiş iş ya da satır bulunamadı")
        log_access(who, "satir_silindi", job_id, request, f"satır {idx}")
        return {"index": idx, "silindi": True}

    @api.post("/webhooks/test", dependencies=[auth])
    def webhook_test(body: WebhookTest):
        """Entegrasyon denemesi: verilen adrese imzalı 'ping' gönderir, sonucu döndürür."""
        problem = callback_problem(body.url, settings)
        if problem:
            raise HTTPException(422, problem)
        from viet_asr.infrastructure.webhook import send_test

        return send_test(body.url, settings.webhook_secret)

    @api.get("/webhooks/dead", dependencies=[auth])
    def webhooks_dead(limit: int = 100):
        """Deneme hakkı biten (ölü) teslimler."""
        return [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in d.items()
                 if k != "payload"} for d in store.dead_deliveries(settings.webhook_max_attempts,
                                                                    max(1, min(limit, 1000)))]

    @api.post("/webhooks/{delivery_id}/retry", dependencies=[auth])
    def webhook_retry(delivery_id: int):
        if not store.retry_delivery(delivery_id):
            raise HTTPException(404, "gönderilmemiş teslim bulunamadı")
        return {"id": delivery_id, "yeniden_denenecek": True}

    @api.get("/corrections")
    def list_corrections(request: Request, who: str = auth, limit: int = 1000):
        log_access(who, "duzeltmeler_alindi", None, request, f"en fazla {limit}", dedupe=True)
        return [{"job_id": c.job_id, "index": c.idx, "language": c.language,
                 "start": round(c.start, 3), "end": round(c.end, 3), "speaker": c.speaker,
                 "text": c.text, "original_speaker": c.original_speaker,
                 "original_text": c.original_text, "edited_at": c.edited_at.isoformat(),
                 "audio_available": bool(c.audio_path)}
                for c in store.corrections(max(1, min(limit, 10_000)))]

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str, request: Request, who: str = auth):
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(404, "iş bulunamadı")
        tr = store.get_transcript(job_id) if rec.status == JobStatus.DONE else None
        if tr is not None:  # içerik (konuşma metni) gösterildi
            log_access(who, "sonuc_goruntulendi", job_id, request, dedupe=True)
        out = record_to_dict(rec, tr)
        if rec.status == JobStatus.QUEUED:
            out["sira"] = store.queue_position(job_id)  # önündeki bekleyen iş sayısı
        return out

    @api.get("/health")
    def health():
        checks, ok = {}, True
        for name, fn in (health_checks or {}).items():
            try:
                fn()
                checks[name] = "ok"
            except Exception as e:
                checks[name], ok = f"hata: {type(e).__name__}", False
        body = {"status": "ok" if ok else "bozuk", "checks": checks,
                "languages": settings.languages,
                "features": {"dataset_export": settings.enable_dataset_export, "streams": streams is not None and settings.enable_streams}}
        try:
            body["jobs"] = store.counts()
        except Exception:
            pass
        from viet_asr.interface.alerts import collect

        body["uyarilar"] = collect(store, settings)  # yalnızca sayı ve süre, kişisel veri yok
        return JSONResponse(body, status_code=200 if ok else 503)

    @api.get("/health/ready")
    def ready():
        """Hazır olma: veritabanı/kuyruk erişimi ve modellerin yüklü, işçilerin çalışır olması.
        Yük dengeleyici ya da Docker bu 200 dönmeden trafiği bu kopyaya göndermemeli."""
        checks, ok = {}, True
        for name, fn in (health_checks or {}).items():
            try:
                fn()
                checks[name] = "ok"
            except Exception as e:
                checks[name], ok = f"hata: {type(e).__name__}", False
        for name, fn in (readiness or {}).items():
            try:
                good, detail = fn()
            except Exception as e:
                good, detail = False, f"hata: {type(e).__name__}"
            checks[name] = detail
            ok = ok and good
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    @api.get("/audit")
    def audit_log(request: Request, who: str = auth, is_id: str | None = None, kim: str | None = None,
                  limit: int = 200):
        """Erişim kaydı (en yenisi önce). AUDIT_READERS doluysa yalnızca o kişiler okuyabilir."""
        if settings.audit_readers and who.lower() not in settings.audit_readers:
            raise HTTPException(403, "erişim kaydını okuma yetkiniz yok (AUDIT_READERS)")
        log_access(who, "erisim_kaydi_okundu", is_id, request, dedupe=True)
        return store.audit_rows(is_id, kim, max(1, min(limit, 5000)))

    @api.get("/quality", dependencies=[auth])
    def quality(gun: int = 90):
        """Düzeltmelerden kalite: hafta / dil / model başına düzeltme oranı (yaklaşık WER).
        Yalnızca elden geçirilmiş (en az bir satırı düzeltilmiş ya da silinmiş) işler sayılır."""
        from datetime import datetime, timedelta, timezone

        from viet_asr.application.quality import quality_report

        since = datetime.now(timezone.utc) - timedelta(days=max(1, min(gun, 3650)))
        return {"gun": gun, "satirlar": quality_report(store.review_lines(since))}

    @api.get("/finetune/durum", dependencies=[auth])
    def finetune_status():
        """İnce ayar hazırlığı: dil başına veri setine aktarma onayı verilmiş, eğitime uygun uzunlukta
        (1-20 sn) düzeltilmiş satır sayısı ve süresi; eşik: scripts/ince_ayar_dongusu.py ile aynı
        (500 satır, 1 saat; sezgiseldir). Eğitimin kendisi arayüzde yapılmaz (WSL2 + GPU, finetune/)."""
        out = {}
        for c in store.corrections(consented_only=True, use="egitim"):
            if not 1.0 <= c.end - c.start <= 20.0:
                continue
            d = out.setdefault(c.language or "?", {"satir": 0, "sure_sn": 0.0})
            d["satir"] += 1
            d["sure_sn"] = round(d["sure_sn"] + (c.end - c.start), 1)
        for d in out.values():
            d["esik_asildi"] = d["satir"] >= 500 and d["sure_sn"] >= 3600
        return {"esik_satir": 500, "esik_sure_sn": 3600, "veri_seti_acik": settings.enable_dataset_export,
                "diller": out}

    @api.get("/metrics/prometheus", dependencies=[auth], response_class=PlainTextResponse)
    def metrics_prometheus():
        """Prometheus biçiminde: iş sayıları, kuyruk derinliği ve en eski mesaj yaşı, aşama süre
        histogramları, gerçek zaman oranı, webhook teslimleri, DLQ."""
        from viet_asr.interface.metrics import prometheus_text

        return PlainTextResponse(prometheus_text(store, settings, queue_stats),
                                 media_type="text/plain; version=0.0.4; charset=utf-8")

    @api.get("/metrics", dependencies=[auth])
    def metrics():
        """Son 500 biten iş: istekten bitişe süre ve 'bitiş süresi / ses süresi' oranı.
        Hedef: oran_p95 <= 1 (1 dk ses en geç 1 dk'da biter)."""
        body = {"gecikme": latency_stats(store.recent_done(500)), "isler": store.counts()}
        if queue_stats is not None:
            try:
                body["kuyruk"] = queue_stats()
            except Exception as e:
                body["kuyruk"] = f"hata: {type(e).__name__}"
        return body

    return api


def _production_app() -> FastAPI:
    import redis

    from viet_asr.interface.composition import store_from_settings
    from viet_asr.infrastructure.redis_streams import RedisStreamJobQueue

    s = Settings()
    store = store_from_settings(s)
    store.migrate()
    r = redis.Redis.from_url(s.redis_url)
    queue = RedisStreamJobQueue(r)
    limiter = None
    if s.rate_limit_per_minute > 0:  # birden çok API süreci aynı sayacı görsün
        from viet_asr.infrastructure.rate_limit import RedisRateLimiter

        limiter = RedisRateLimiter(r, s.rate_limit_per_minute)
    def workers():
        alive = {q: queue.live_consumers(q) for q in ["lid", *s.languages]}
        missing = [q for q, n in alive.items() if n == 0]
        return not missing, (f"çalışan işçi yok: {missing}" if missing else
                             ", ".join(f"{q}={n}" for q, n in alive.items()))

    # Canlı altyazı ve canlı akış bu süreçte çözülür (gecikme için kuyruğa girmez); modeller ilk
    # oturumda yüklenir. Oturum bitince kayıt kuyruğa: konuşmacı ayrımını işçiler yapar.
    engines = resolver = cfg = None
    if s.enable_live or s.enable_streams:
        from viet_asr.interface.composition import build_engines, build_resolver, config_from_settings

        cfg = config_from_settings(s)
        engines, resolver = build_engines(cfg), build_resolver(cfg)
    from viet_asr.application.jobs import SubmitJob

    submit = SubmitJob(store, queue, s.languages)
    streams = None
    if s.enable_streams:
        from viet_asr.interface.streams import StreamManager

        streams = StreamManager.from_settings(s, engines, cfg.vad, submit)
    app = create_app(store, queue, s, {"veritabani": store.ping, "kuyruk": r.ping},
                     queue_stats=lambda: queue.stats(["lid", *s.languages]), rate_limiter=limiter,
                     readiness={"isciler": workers}, streams=streams)
    app.state.streams = streams
    if s.enable_live:
        from viet_asr.interface.composition import wire_live

        wire_live(app, s, cfg, engines, resolver, store, submit, warm=False)
    return app


def main(argv=None):
    """Tam mod API'si: python -m viet_asr.interface.api [--host --port] (HTTPS=1 destekli)."""
    import argparse
    import logging

    from viet_asr.interface.serve import run

    s = Settings()
    p = argparse.ArgumentParser(description="Tam mod API sunucusu")
    p.add_argument("--host", default=s.host)
    p.add_argument("--port", type=int, default=s.port)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(_production_app(), s, a.host, a.port)


def __getattr__(name):  # uvicorn viet_asr.interface.api:app -> ilk erişimde kurulur
    if name == "app":
        globals()["app"] = _production_app()
        return globals()["app"]
    raise AttributeError(name)


if __name__ == "__main__":
    main()
