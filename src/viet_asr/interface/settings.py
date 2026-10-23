"""Ortam değişkenlerinden ayarlar.

Sıra: gerçek ortam değişkeni > proje kökündeki .env dosyası > buradaki varsayılan.
Windows'ta baslat.bat ilk çalıştırmada .env.example'dan .env oluşturur.
"""
from __future__ import annotations

import os

try:  # .env varsa yükle (python-dotenv yoksa sessizce geç)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass
from dataclasses import dataclass, field
from pathlib import Path


def _list(v: str) -> list[str]:
    return [x.strip().lower() for x in v.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # development: API anahtarı ve http/özel ağ callback'i serbest; production: zorunlu/kısıtlı
    app_env: str = field(default_factory=lambda: os.environ.get("APP_ENV", "development").lower())
    host: str = field(default_factory=lambda: os.environ.get("HOST", "127.0.0.1"))
    # Başka cihazdan (telefon, diğer bilgisayar) mikrofon: tarayıcılar mikrofonu yalnızca https'de
    # açar. HTTPS=1: sertifika yoksa kendinden imzalı üretilir (DATA_DIR/tls). HOST=0.0.0.0 da gerekir.
    https: bool = field(default_factory=lambda: os.environ.get("HTTPS", "0").lower() in ("1", "true", "evet"))
    tls_cert: str = field(default_factory=lambda: os.environ.get("TLS_CERT", ""))
    tls_key: str = field(default_factory=lambda: os.environ.get("TLS_KEY", ""))
    tls_dir: str = field(default_factory=lambda: os.environ.get("TLS_DIR", "data/tls"))
    # Canlı altyazı (/canli, /ws/live, WebRTC). Tam modda modeller API sürecinde ilk oturumda yüklenir.
    enable_live: bool = field(default_factory=lambda: os.environ.get(
        "ENABLE_LIVE", "1").lower() in ("1", "true", "evet"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8001")))
    api_key: str = field(default_factory=lambda: os.environ.get("API_KEY", "").strip())
    # Kişi başına anahtar (erişim kaydında "kim" yazsın): "Tuğba:anahtar1,Ayşe:anahtar2".
    # API_KEY de geçerli kalır (adı "sistem": entegre olan sistem). Anahtarda ':' ve ',' olmasın.
    api_keys_setting: str = field(default_factory=lambda: os.environ.get("API_KEYS", "").strip())
    # Erişim kaydını (GET /audit) yalnızca bu kişiler okuyabilir ("Tuğba,Ayşe"); boşsa anahtarı olan herkes
    audit_readers: list[str] = field(default_factory=lambda: _list(os.environ.get("AUDIT_READERS", "")))
    # Erişim kaydı kaç gün saklansın (iş silinse de kayıt kalır; KVKK denetimi için uzun)
    audit_retention_days: float = field(default_factory=lambda: float(os.environ.get(
        "AUDIT_RETENTION_DAYS", "730")))
    webhook_secret: str = field(default_factory=lambda: os.environ.get("WEBHOOK_SECRET", "").strip())
    # boş değilse callback yalnızca bu adreslere gidebilir (iç ağ adresleri de yazılabilir)
    webhook_allowed_hosts: list[str] = field(default_factory=lambda: _list(
        os.environ.get("WEBHOOK_ALLOWED_HOSTS", "")))
    database_url: str = field(default_factory=lambda: os.environ.get(
        "DATABASE_URL", "postgresql://asr:asr@localhost:5432/asr"))
    # yerel mod (baslat.bat) veritabanı; boşsa DATA_DIR içinde jobs.db (SQLite).
    # PostgreSQL de verilebilir: LOCAL_DATABASE_URL=postgresql://...
    local_database_url: str = field(default_factory=lambda: os.environ.get("LOCAL_DATABASE_URL", ""))
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    models_dir: str = field(default_factory=lambda: os.environ.get("MODELS_DIR", "models"))
    data_dir: str = field(default_factory=lambda: os.environ.get("DATA_DIR", "data/uploads"))
    languages: list[str] = field(default_factory=lambda: _list(os.environ.get("LANGUAGES", "vi,tr")))
    vi_model: str | None = field(default_factory=lambda: os.environ.get("VI_MODEL") or None)
    tr_model: str | None = field(default_factory=lambda: os.environ.get("TR_MODEL") or None)
    whisper_device: str = field(default_factory=lambda: os.environ.get("WHISPER_DEVICE", "auto"))
    whisper_compute_type: str = field(default_factory=lambda: os.environ.get("WHISPER_COMPUTE_TYPE", "auto"))
    whisper_hotwords: str = field(default_factory=lambda: os.environ.get("WHISPER_HOTWORDS", "").strip())
    whisper_initial_prompt: str = field(default_factory=lambda: os.environ.get("WHISPER_INITIAL_PROMPT", "").strip())
    threads: int = field(default_factory=lambda: int(os.environ.get("ONNX_THREADS", "2")))
    # faster-whisper (Türkçe) bir çözüm için iş parçacığı; "auto" = fiziksel çekirdek (2-8)
    whisper_threads_raw: str = field(default_factory=lambda: os.environ.get("WHISPER_THREADS", "auto"))
    whisper_beam_size: int = field(default_factory=lambda: int(os.environ.get("WHISPER_BEAM_SIZE", "5")))
    whisper_batch_size: int = field(default_factory=lambda: int(os.environ.get("WHISPER_BATCH_SIZE", "8")))
    # 0: konuşmacı sayısı sesten bulunur (istekte num_speakers verilirse o kullanılır)
    speakers: int = field(default_factory=lambda: int(os.environ.get("NUM_SPEAKERS", "0")))
    diarizer: str = field(default_factory=lambda: os.environ.get("DIARIZER", "cluster"))
    diarizer_window: float = field(default_factory=lambda: float(os.environ.get("DIARIZER_WINDOW", "1.0")))
    diarizer_hop: float = field(default_factory=lambda: float(os.environ.get("DIARIZER_HOP", "0.4")))
    diarizer_min_window: float = field(default_factory=lambda: float(os.environ.get("DIARIZER_MIN_WINDOW", "0.4")))
    diarizer_merge_distance: float = field(default_factory=lambda: float(os.environ.get("DIARIZER_MERGE_DISTANCE", "0.35")))
    hotwords: str | None = field(default_factory=lambda: os.environ.get("VI_HOTWORDS") or None)
    max_audio_minutes: float = field(default_factory=lambda: float(os.environ.get("MAX_AUDIO_MINUTES", "60")))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB", "100")))
    # transkript çıktısında cümle başı büyük harf ve cümle sonu noktalama (. / ?)
    punctuate: bool = field(default_factory=lambda: os.environ.get("PUNCTUATE", "0") == "1")
    # stüdyo diyaloglarında sunucu/katılımcı rollerini çözümleme
    detect_roles: bool = field(default_factory=lambda: os.environ.get("DETECT_ROLES", "0") == "1")
    # aynı kelimenin art arda tekrarı ve dolgu sesleri temizliği (varsayılan kapalı)
    clean_repeats: bool = field(default_factory=lambda: os.environ.get("CLEAN_REPEATS", "0") == "1")
    # --- Canlı altyazı (/canli, ws://.../ws/live; Live-stream'den) ---
    # Canlı sayfanın varsayılan dili: tr | vi | auto (auto: ilk cümleden bulunur ve kilitlenir)
    live_language: str = field(default_factory=lambda: os.environ.get("LIVE_LANGUAGE", "vi").lower())
    # Canlı modun Türkçe modeli; boşsa TR_MODEL (model bir kez yüklenir). Vietnamca: hep VI_MODEL.
    live_model: str | None = field(default_factory=lambda: os.environ.get("LIVE_MODEL") or None)
    live_vad: str = field(default_factory=lambda: os.environ.get("LIVE_VAD", "silero").lower())
    live_vad_threshold: float = field(default_factory=lambda: float(os.environ.get(
        "LIVE_VAD_THRESHOLD", "0.5")))
    # Türkçe (Whisper): bu kadar duraklama cümleyi bitirir; durmadan konuşulursa en fazla bu kadar sn
    end_silence_ms: int = field(default_factory=lambda: int(os.environ.get("END_SILENCE_MS", "700")))
    max_utterance_sec: float = field(default_factory=lambda: float(os.environ.get(
        "MAX_UTTERANCE_SEC", "12")))
    partial_interval_sec: float = field(default_factory=lambda: float(os.environ.get(
        "PARTIAL_INTERVAL_SEC", "1.2")))
    save_recordings: bool = field(default_factory=lambda: os.environ.get(
        "SAVE_RECORDINGS", "true").lower() in ("1", "true", "evet"))
    recordings_dir: str = field(default_factory=lambda: os.environ.get("RECORDINGS_DIR", "recordings"))
    # oturum bitince 'live.ended' bildirimi (sonra aynı adrese konuşmacılı 'job.done'); boşsa yok
    live_webhook_url: str = field(default_factory=lambda: os.environ.get("LIVE_WEBHOOK_URL", "").strip())
    live_speakers: bool = field(default_factory=lambda: os.environ.get(
        "LIVE_SPEAKERS", "true").lower() in ("1", "true", "evet"))
    live_speaker_threshold: float = field(default_factory=lambda: float(os.environ.get(
        "LIVE_SPEAKER_THRESHOLD", "0.5")))
    # bağlantı koparsa oturum bu kadar sn bekler; sayfa dönerse aynı oturum sürer (0: beklemez)
    live_resume_sec: float = field(default_factory=lambda: float(os.environ.get("LIVE_RESUME_SEC", "30")))
    # canlı oturum kayıtları (.wav, .txt, .jsonl) kaç gün sonra silinsin (0 = silme); kişisel veri
    recordings_retention_days: float = field(default_factory=lambda: float(os.environ.get(
        "RECORDINGS_RETENTION_DAYS", "30")))
    # Uyarılar (/health "uyarilar" ve arayüzdeki şerit): kuyrukta bu kadar dk'dan uzun bekleyen iş,
    # diskte bu kadar GB'tan az boş yer
    alert_queue_wait_min: float = field(default_factory=lambda: float(os.environ.get("ALERT_QUEUE_WAIT_MIN", "30")))
    alert_disk_gb: float = field(default_factory=lambda: float(os.environ.get("ALERT_DISK_GB", "5")))
    # Veritabanı yedeği (işler, satırlar, düzeltmeler): bakım her turda son yedeğin yaşına bakar,
    # BACKUP_EVERY_HOURS'u geçtiyse yeni yedek alır (bilgisayar gece kapalı olsa da açılınca alınır).
    backup_every_hours: float = field(default_factory=lambda: float(os.environ.get("BACKUP_EVERY_HOURS", "24")))
    backup_keep_days: float = field(default_factory=lambda: float(os.environ.get("BACKUP_KEEP_DAYS", "14")))
    backup_dir_setting: str = field(default_factory=lambda: os.environ.get("BACKUP_DIR", ""))
    # PostgreSQL Docker'daysa pg_dump konteynerin içinden çalışır (Windows'a kurulum gerekmez)
    backup_docker_container: str = field(default_factory=lambda: os.environ.get(
        "BACKUP_DOCKER_CONTAINER", "asr-pg"))
    pg_dump: str = field(default_factory=lambda: os.environ.get("PG_DUMP", ""))
    live_auto_analyze: bool = field(default_factory=lambda: os.environ.get(
        "LIVE_AUTO_ANALYZE", "true").lower() in ("1", "true", "evet"))
    # canlı akış (RTP / ffmpeg kaynağı; POST /streams). Varsayılan: geliştirmede açık, üretimde kapalı
    # (istek bir UDP portu açar ya da dış adrese bağlanır).
    enable_streams: bool = field(default_factory=lambda: os.environ.get(
        "ENABLE_STREAMS", "0" if os.environ.get("APP_ENV", "development").lower() == "production" else "1") == "1")
    stream_max_sessions: int = field(default_factory=lambda: int(os.environ.get("STREAM_MAX_SESSIONS", "4")))
    # POST /streams "rtp://0.0.0.0:0" için UDP port aralığı ("40000-40100"; boşsa işletim sistemi seçer) ve
    # karşı servise söylenecek bu sunucunun adresi (yanıttaki "rtp_adres"). Docker/güvenlik duvarında
    # yalnızca bu aralık UDP için açılır.
    stream_rtp_ports: str = field(default_factory=lambda: os.environ.get("STREAM_RTP_PORTS", "").strip())
    # dışarıdan çekilen akış kaynakları (http, rtsp, rtmp, srt) yalnızca bu adreslere (iç ağ olabilir);
    # boşsa production'da iç ağ adresleri reddedilir (SSRF)
    stream_allowed_hosts: list[str] = field(default_factory=lambda: _list(os.environ.get("STREAM_ALLOWED_HOSTS", "")))
    # push akışında işlenmeyi bekleyebilecek en çok ses (sn); aşılınca 429 (gönderen yavaşlar)
    stream_max_backlog: float = field(default_factory=lambda: float(os.environ.get("STREAM_MAX_BACKLOG", "30")))
    stream_public_host: str = field(default_factory=lambda: os.environ.get("STREAM_PUBLIC_HOST", "").strip())
    # canlı akışta bir parça en fazla bu kadar sürer (sn): uzun konuşmada gecikmenin üst sınırı
    stream_max_speech: float = field(default_factory=lambda: float(os.environ.get("STREAM_MAX_SPEECH", "10")))
    # sunucudaki bir dosya yolunu doğrudan kabul et (toplu iş için; yalnızca güvenilir ağda aç)
    allow_local_paths: bool = field(default_factory=lambda: os.environ.get(
        "ALLOW_LOCAL_PATHS", "0") == "1")
    # iki aşamalı canlı akış: hızlı modelle (seda-v0.1, akışlı) anlık metin, medium ile arka planda düzeltme
    stream_two_stage: bool = field(default_factory=lambda: os.environ.get(
        "STREAM_TWO_STAGE", "0").lower() in ("1", "true", "evet"))
    stream_fast_model: str = field(default_factory=lambda: os.environ.get(
        "STREAM_FAST_MODEL", "seda-v0.1"))
    stream_two_stage_max: int = field(default_factory=lambda: int(os.environ.get("STREAM_TWO_STAGE_MAX", "1")))
    # worker açılırken önceden yüklenecek diller (ilk işte gecikme olmasın)
    warmup: list[str] = field(default_factory=lambda: _list(os.environ.get("WARMUP_LANGS", "")))
    # dil tespiti worker'ı: Whisper-small'ı da baştan yükle (yoksa ilk zor seste ~5-15 sn gecikme)

    # saklama: biten işlerin ses dosyası N saat, iş kaydı ve sonucu N gün sonra silinir (0 = silme)
    audio_retention_hours: int = field(default_factory=lambda: int(os.environ.get(
        "AUDIO_RETENTION_HOURS", "24")))
    result_retention_days: int = field(default_factory=lambda: int(os.environ.get(
        "RESULT_RETENTION_DAYS", "30")))
    # RUNNING'de bundan uzun kalan iş askıda sayılır; MAX_JOB_ATTEMPTS denemeden sonra FAILED
    processing_stale_sec: int = field(default_factory=lambda: int(os.environ.get(
        "PROCESSING_STALE_SEC", "1800")))
    # Aşırı yük koruması. Kuyrukta/işlemde bekleyen iş bu sayıyı aşınca yeni istek 503 +
    # Retry-After alır (sistem çökmeden "sonra tekrar dene" der). 0 = sınırsız.
    # Varsayılan 10000: beklenen 4000 mesajlık ani yükün 2,5 katı.
    max_pending_jobs: int = field(default_factory=lambda: int(os.environ.get("MAX_PENDING_JOBS", "10000")))
    # İstemci (API anahtarı ya da IP) başına dakikada en fazla istek; 0 = kapalı (varsayılan).
    # Kapalı, çünkü tek entegre sistem 4000 mesajı bir anda gönderebilir; açılırsa 429 alır.
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.environ.get(
        "RATE_LIMIT_PER_MINUTE", "0")))
    # İşlem profili (istekte profile= ile iş başına değiştirilebilir): full | feedback
    pipeline_profile: str = field(default_factory=lambda: os.environ.get("PIPELINE_PROFILE", "full"))
    # feedback profilinde bundan kısa seste konuşmacı ayırma atlanır (sn)
    pipeline_min_diarize_sec: float = field(default_factory=lambda: float(os.environ.get(
        "PIPELINE_MIN_DIARIZE_SEC", "10")))
    # araya sıkışmış kısa konuşmacıyı komşusuna ver (sn; 0 = kapalı). Bkz. FusionConfig.
    fusion_sandwich_max: float = field(default_factory=lambda: float(os.environ.get(
        "FUSION_SANDWICH_MAX", "0")))
    # callback'e hangi olaylarda bildirim gitsin (istekte notify= ile iş başına)
    notify_default: list[str] = field(default_factory=lambda: _list(os.environ.get(
        "NOTIFY_EVENTS", "done,failed,rejected")))
    # webhook teslimi en fazla bu kadar denenir (10 sn'den 1 saate artan aralıklarla), sonra "ölü"
    webhook_max_attempts: int = field(default_factory=lambda: int(os.environ.get(
        "WEBHOOK_MAX_ATTEMPTS", "8")))
    # QUEUED'da bundan uzun kalan iş, kuyruğu boşsa yeniden kuyruğa konur (tam mod, sn)
    queued_stale_sec: int = field(default_factory=lambda: int(os.environ.get("QUEUED_STALE_SEC", "300")))
    # Kontrol edilmiş sonuçları veri setine (test/eğitim) aktarma; varsayılan kapalı. Açıkken bile
    # her aktarım kontrol eden kişinin adı ve açık onay ister; önce bekleyen kuyruğa düşer.
    # temiz transkriptte konuşmacı adları: genel (Konuşmacı 1) | studyo (Sunucu, Katılımcı 1) | yok
    transcript_labels: str = field(default_factory=lambda: os.environ.get("TRANSCRIPT_LABELS", "genel"))
    enable_dataset_export: bool = field(default_factory=lambda: os.environ.get(
        "ENABLE_DATASET_EXPORT", "0").lower() in ("1", "true", "evet"))
    dataset_dir: str = field(default_factory=lambda: os.environ.get("DATASET_DIR", "data/veri_seti"))
    max_job_attempts: int = field(default_factory=lambda: int(os.environ.get("MAX_JOB_ATTEMPTS", "3")))
    # modeller yüklenirken MANIFEST.json'daki sha256 ile karşılaştır
    verify_models: bool = field(default_factory=lambda: os.environ.get("VERIFY_MODELS", "1") == "1")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def api_keys(self) -> dict[str, str]:
        """anahtar -> kişi adı (API_KEY "sistem" adıyla dahil)."""
        out = {self.api_key: "sistem"} if self.api_key else {}
        for item in self.api_keys_setting.split(","):
            name, sep, key = item.strip().rpartition(":")
            if sep and name.strip() and key.strip():
                out[key.strip()] = name.strip()
        return out

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    @property
    def backup_dir(self) -> str:
        """Boşsa yükleme klasörünün yanında: data/uploads -> data/yedek."""
        return self.backup_dir_setting or str(Path(self.data_dir).parent / "yedek")

    @property
    def local_db(self) -> str:
        return self.local_database_url or f"sqlite:///{Path(self.data_dir) / 'jobs.db'}"

    @property
    def concurrency(self) -> int:
        """Yerel mod (tek süreç, iş parçacıkları) için aynı anda işlenen ses sayısı."""
        return self.concurrency_plan()[0]

    @property
    def whisper_threads(self) -> int:
        """Whisper bir çözümde kaç iş parçacığı kullansın. Tek iş geldiğinde de işlemci boş
        kalmasın diye ONNX modellerinden (2) ayrı: varsayılan fiziksel çekirdek sayısı."""
        if self.whisper_threads_raw != "auto":
            return max(1, int(self.whisper_threads_raw))
        return max(2, min(8, physical_cpus()))

    def whisper_workers(self, jobs: int) -> int:
        """Aynı anda kaç Whisper çözümü: 1. FasterWhisperAsr modeli kilitle korur (Windows'ta
        eşzamanlı çağrıda CTranslate2 çökmesi), yani çözümler zaten sırayla; CTranslate2'ye 2 işçi
        vermek yalnızca modelin ikinci kopyasına bellek harcardı. Tek çözüm WHISPER_THREADS
        (fiziksel çekirdek) iş parçacığıyla çalışır; fazla Türkçe işler sırada bekler."""
        return 1

    def model_for(self, lang: str) -> str:
        from viet_asr.infrastructure.catalog import DEFAULT_MODELS

        return self.models.get(lang) or DEFAULT_MODELS.get(lang, "zipformer-68m")

    def concurrency_plan(self, queue: str | None = None) -> tuple[int, str]:
        """(iş sayısı, gerekçe). CONCURRENCY sayıysa o; 'auto' (varsayılan) ise çekirdek ve
        belleğin küçüğü.

        queue=None: yerel mod — tek süreç, modeller bir kez yüklenir, işler paylaşır.
        queue='vi'|'tr': tam mod — her süreç kendi modelini yükler (stream_worker --surec).
        Bellek: toplam belleğin %75'i (işletim sistemi ve diğer programlara pay bırakılır).
        """
        from viet_asr.infrastructure.catalog import AUX_MEMORY_GB, memory_estimate

        v = os.environ.get("CONCURRENCY", "auto")
        if v != "auto":
            return max(1, int(v)), "CONCURRENCY ayarı"
        per = self.whisper_threads if queue and self.model_for(queue).startswith("faster-whisper") \
            else self.threads
        by_cpu = max(1, available_cpus() // max(1, per))
        total = total_memory_gb()
        if total is None:
            return by_cpu, f"çekirdek: {available_cpus()} / {self.threads} (bellek okunamadı)"
        usable = total * 0.75
        langs = [queue] if queue else list(self.languages)
        est = {lang: memory_estimate(self.model_for(lang)) for lang in langs}
        if queue:  # her süreç: model + yardımcı modeller + bir iş
            fixed, per_job = 0.0, est[queue][0] + est[queue][1] + AUX_MEMORY_GB
        else:      # tek süreç: tüm modeller bir kez, her eşzamanlı iş en ağır dilin çalışma belleği
            fixed = AUX_MEMORY_GB + sum(f for f, _ in est.values())
            per_job = max(p for _, p in est.values())
        by_mem = max(1, int((usable - fixed) // per_job))
        n = min(by_cpu, by_mem)
        why = (f"çekirdek {available_cpus()}/{self.threads}={by_cpu}, bellek {total:.0f} GB "
               f"(%75: {usable:.1f}, sabit {fixed:.1f} + iş başı {per_job:.1f}) = {by_mem}")
        return n, why

    @property
    def models(self) -> dict[str, str]:
        return {k: v for k, v in (("vi", self.vi_model), ("tr", self.tr_model)) if v}


def total_memory_gb() -> float | None:
    """Bu sürecin kullanabileceği toplam bellek (GB): Docker sınırı, yoksa makinenin belleği."""
    try:
        v = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if v != "max":
            return int(v) / 2**30
    except (OSError, ValueError):
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    if os.name == "nt":  # psutil yoksa Windows API'si
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys / 2**30
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def physical_cpus() -> int:
    """Fiziksel çekirdek (hiper iş parçacığı sayılmaz: int8 matris çarpımında ikinci iş
    parçacığı az hız katar). psutil yoksa mantıksal çekirdeğin yarısı."""
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return min(n, available_cpus())
    except ImportError:
        pass
    return max(1, available_cpus() // 2)


def available_cpus() -> int:
    """Bu sürecin gerçekten kullanabileceği çekirdek sayısı.

    os.cpu_count() Docker'da ana makinenin TÜM çekirdeklerini döndürür; konteynere
    sınır konduysa (cpus: 4) cgroup'tan okunur, yoksa sürecin izinli çekirdekleri.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS
        return os.cpu_count() or 1
