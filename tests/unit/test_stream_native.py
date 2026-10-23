"""Gerçek akışlı model yolu: ses geldikçe ara metin, cümle sonunda kesin parça, düzeltme motoruna ses."""
import numpy as np

from viet_asr.application.stream import StreamTranscriber
from viet_asr.domain import Word

SR = 16000


class FakeHandle:
    """Her 0,5 sn'lik konuşmaya bir kelime ekler; 0,6 sn sessizlik gelince cümle biter."""

    def __init__(self):
        self.speech, self.silence, self.words = 0, 0, []

    def accept(self, frame):
        loud = float(np.abs(frame).max()) > 0.01
        if loud:
            self.silence = 0
            self.speech += len(frame)
            while len(self.words) < self.speech // (SR // 2):
                self.words.append(f"k{len(self.words)}")
        else:
            self.silence += len(frame)

    def text(self):
        return " ".join(self.words)

    def is_endpoint(self):
        return bool(self.words) and self.silence >= int(0.6 * SR)

    def reset(self):
        self.speech, self.silence, self.words = 0, 0, []

    def finish(self):
        return self.text()


class FakeStreamingAsr:
    name = "seda-sahte"
    segmentation = "full"
    streaming = True
    max_utterance = 12.0

    def open_stream(self):
        return FakeHandle()

    def words_for(self, text, start, end):
        parts = text.split()
        d = (end - start) / max(1, len(parts))
        return [Word(text=p, start=start + i * d, end=start + (i + 1) * d) for i, p in enumerate(parts)]


class ListSource:
    def __init__(self, frames):
        self.frames_, self.closed = frames, False

    def frames(self):
        yield from self.frames_

    def close(self):
        self.closed = True


class RecordingRefiner:
    name = "refiner"
    segmentation = "vad"

    def __init__(self):
        self.calls = []

    def transcribe(self, segments):
        self.calls.append([(s.start, len(s.samples)) for s in segments])
        return [Word(text="düzeltildi", start=segments[0].start, end=segments[0].start + 0.5)]


def _frames(pattern):
    """pattern: [(saniye, yüksek_mi)] -> 0,1 sn'lik parçalar."""
    out = []
    for dur, loud in pattern:
        for _ in range(int(round(dur * 10))):
            out.append(np.full(SR // 10, 0.3 if loud else 0.0, np.float32))
    return out


def test_ara_metin_buyur_cumle_sonunda_kesin_parca_cikar():
    events = []
    st = StreamTranscriber(lambda **k: None)
    sm = st.run(ListSource(_frames([(2.0, True), (1.0, False), (1.5, True), (1.0, False)])),
                FakeStreamingAsr(), events.append, interim=True)
    finals = [e for e in events if e.final]
    interims = [e for e in events if not e.final]
    assert [e.text for e in finals] == ["k0 k1 k2 k3", "k0 k1 k2"]
    assert [e.index for e in finals] == [0, 1]
    assert interims and interims[0].text == "k0"          # ilk kelime 0,5 sn konuşmadan sonra görünür
    assert sm.segments == 2 and sm.words == 7
    assert all(0.0 <= e.latency < 1.0 for e in finals)


def test_duzeltme_motoru_kesin_parcanin_sesini_alir():
    events, ref = [], RecordingRefiner()
    st = StreamTranscriber(lambda **k: None)
    st.run(ListSource(_frames([(2.0, True), (1.0, False), (1.5, True), (1.0, False)])),
           FakeStreamingAsr(), events.append, interim=True, refiner=ref)
    assert len(ref.calls) >= 1
    n = sum(len(c) for call in ref.calls for c in [call])
    assert n == 2                                        # iki kesin parça düzeltmeye gitti
    assert any(e.duzeltildi for e in events)


def test_akis_bitince_acik_cumle_kesin_olur_ve_ara_metin_kapaliysa_yok():
    events = []
    st = StreamTranscriber(lambda **k: None)
    st.run(ListSource(_frames([(1.2, True)])), FakeStreamingAsr(), events.append, interim=False)
    assert [e.text for e in events] == ["k0 k1"] and events[0].final


def test_uzun_cumle_nefeste_kesilir_kelime_ortasinda_degil():
    """12 sn sınırına varmadan, yumuşak sınırdan sonraki ilk kısa sessizlikte parça kesilir."""
    import numpy as np
    from viet_asr.application.stream import StreamTranscriber

    SRr = 16000
    sesli = np.full(SRr // 2, 0.3, np.float32)       # 0.5 sn yüksek ses
    sessiz = np.zeros(SRr // 2, np.float32)          # 0.5 sn sessizlik
    kareler = []
    for _ in range(14):                              # ~7 sn konuşma + kısa aralar
        kareler += [sesli[i:i + 320] for i in range(0, len(sesli), 320)]
        kareler += [sessiz[i:i + 160] for i in range(0, 160 * 2, 160)]  # 0.02 sn: nefes değil
    kareler += [sesli[i:i + 320] for i in range(0, len(sesli), 320)]
    kareler += [sessiz[i:i + 320] for i in range(0, 320 * 20, 320)]      # 0.4 sn gerçek nefes
    kareler += [sesli[i:i + 320] for i in range(0, len(sesli), 320)]

    class Asr:
        streaming = True
        segmentation = "full"
        max_utterance = 12.0
        soft_utterance = 6.0
        soft_pause = 0.25
        name = "fake"

        def open_stream(self):
            class H:
                n = 0
                def accept(self, f): self.n += 1
                def text(self): return "kelime" if self.n else ""
                def is_endpoint(self): return False
                def reset(self): self.n = 0
                def finish(self): return "kelime"
            return H()

        def words_for(self, text, a, b):
            return []

    class Src:
        def frames(self):
            yield from kareler

        def close(self):
            pass

    out = []
    StreamTranscriber(lambda **k: None).run(Src(), Asr(), out.append, interim=False)
    kesin = [x for x in out if x.final]
    assert len(kesin) >= 2
    # ilk parça zorla kesim sınırına (12 sn) varmadan, nefeste kesildi
    assert kesin[0].end - kesin[0].start < 11.5
