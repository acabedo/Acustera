
# AcusTera acoustic module v2 — jitter, shimmer, CPP added
import numpy as np, warnings
warnings.filterwarnings("ignore")
_lib = None
_pm  = None

def _librosa():
    global _lib
    if _lib is None:
        import librosa as l; _lib = l
    return _lib

def _parsel():
    global _pm
    if _pm is None:
        import parselmouth as p; _pm = p
    return _pm

def extract_segment(path, start, end, n_mfcc=13):
    lib = _librosa()
    dur = max(end - start, 0.1)
    y, sr = lib.load(path, offset=float(start), duration=dur,
                     sr=None, mono=True)
    if len(y) < sr * 0.5:
        return _empty(n_mfcc)
    out = {"duration": dur}

    # F0 + speech_rate via Praat
    try:
        snd = _parsel().Sound(y, sr)
        f0  = snd.to_pitch().selected_array["frequency"]
        f0  = f0[f0 > 0]
        out["f0_mean"]     = float(np.mean(f0))   if len(f0) > 3 else float("nan")
        out["f0_sd"]       = float(np.std(f0))    if len(f0) > 3 else float("nan")
        out["f0_range"]    = float(np.ptp(f0))    if len(f0) > 3 else float("nan")
        out["speech_rate"] = float(len(f0) / dur)
    except:
        out.update({"f0_mean": float("nan"), "f0_sd": float("nan"),
                    "f0_range": float("nan"), "speech_rate": float("nan")})

    # Intensity via Praat
    try:
        iv = _parsel().Sound(y, sr).to_intensity().values[0]
        iv = iv[iv > 0]
        out["intensity_mean"] = float(np.mean(iv)) if len(iv) else float("nan")
        out["intensity_sd"]   = float(np.std(iv))  if len(iv) else float("nan")
    except:
        out["intensity_mean"] = out["intensity_sd"] = float("nan")

    # HNR via Praat
    try:
        hv = _parsel().Sound(y, sr).to_harmonicity().values[0]
        hv = hv[hv > -200]
        out["hnr"] = float(np.mean(hv)) if len(hv) else float("nan")
    except:
        out["hnr"] = float("nan")

    # Local jitter: variation of consecutive F0 periods (pure numpy)
    # Definition: mean(|T[i+1]-T[i]|) / mean(T[i]) (simplified RAP)
    try:
        pitch_obj = _parsel().Sound(y, sr).to_pitch()
        periods = pitch_obj.selected_array["frequency"]
        periods = periods[periods > 0]
        if len(periods) > 3:
            period_s = 1.0 / periods
            diffs = np.abs(np.diff(period_s))
            out["jitter_local"] = float(np.mean(diffs) / np.mean(period_s))
        else:
            out["jitter_local"] = float("nan")
    except Exception:
        out["jitter_local"] = float("nan")

    # Local shimmer: variation in the amplitude of sound peaks (pure NumPy)
    # Aproximacion via energia RMS en ventanas de pitch period
    try:
        pitch_obj2 = _parsel().Sound(y, sr).to_pitch()
        f0_arr = pitch_obj2.selected_array["frequency"]
        f0_arr = f0_arr[f0_arr > 0]
        if len(f0_arr) > 3:
            mean_f0 = float(np.mean(f0_arr))
            win = max(1, int(sr / mean_f0))
            amps = np.array([float(np.sqrt(np.mean(y[i:i+win]**2)))
                             for i in range(0, len(y)-win, win//2)])
            amps = amps[amps > 1e-8]
            if len(amps) > 3:
                diffs_a = np.abs(np.diff(amps))
                out["shimmer_local"] = float(np.mean(diffs_a) / np.mean(amps))
            else:
                out["shimmer_local"] = float("nan")
        else:
            out["shimmer_local"] = float("nan")
    except Exception:
        out["shimmer_local"] = float("nan")

    # CPP via librosa cepstrum (Cepstral Peak Prominence proxy)
    # High values ​​= clear voice; low values ​​= strained or crackling voice
    try:
        import numpy as _np2
        n_fft = min(2048, len(y))
        D  = np.abs(np.fft.rfft(y, n=n_fft))
        D  = _np2.where(D > 0, D, 1e-10)
        ceps = np.real(np.fft.irfft(np.log(D)))
        sr_int = int(sr)
        lo = int(sr_int / 500)
        hi = int(sr_int / 60)
        hi = min(hi, len(ceps) - 1)
        if hi > lo:
            peak = float(np.max(np.abs(ceps[lo:hi])))
            out["cpp"] = float(20 * np.log10(peak + 1e-10))
        else:
            out["cpp"] = float("nan")
    except Exception:
        out["cpp"] = float("nan")

    # MFCCs
    try:
        m = lib.feature.mfcc(y=y, sr=sr, n_mfcc=int(n_mfcc))
        for i in range(int(n_mfcc)):
            out[f"mfcc{i+1}"] = float(np.mean(m[i]))
    except:
        for i in range(int(n_mfcc)): out[f"mfcc{i+1}"] = float("nan")

    # Espectro
    try:
        out["spectral_centroid"] = float(np.mean(lib.feature.spectral_centroid(y=y, sr=sr)))
        out["zcr"]               = float(np.mean(lib.feature.zero_crossing_rate(y)))
        out["rms"]               = float(np.mean(lib.feature.rms(y=y)))
    except:
        out["spectral_centroid"] = out["zcr"] = out["rms"] = float("nan")

    return out

def embedding(path, start, end):
    lib = _librosa()
    dur = max(end - start, 0.1)
    try:
        y, sr = lib.load(path, offset=float(start), duration=dur,
                         sr=None, mono=True)
        m = lib.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        v = np.concatenate([np.mean(m, axis=1), np.std(m, axis=1)])
        return v.tolist()
    except:
        return [0.0] * 26

def cosine_sim(a, b):
    va, vb = np.array(a), np.array(b)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0

def _empty(n_mfcc=13):
    d = {k: float("nan") for k in [
        "duration","f0_mean","f0_sd","f0_range","speech_rate",
        "intensity_mean","intensity_sd","hnr",
        "jitter_local","shimmer_local","cpp",
        "spectral_centroid","zcr","rms"]}
    for i in range(int(n_mfcc)): d[f"mfcc{i+1}"] = float("nan")
    return d

