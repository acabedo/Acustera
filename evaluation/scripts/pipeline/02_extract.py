# Pasos 3a + 5: para cada enunciado >= 2 s
#  - embedding WeSpeaker (pyannote/wespeaker-voxceleb-resnet34-LM) sobre el MP3 16 kHz (primeros 12 s)
#  - prosodia robusta (como pros.py): F0 con rangos masculino (60-300) y femenino (120-500), intensidad
#  - rasgos AcusTera: app_english/acoustic.py::extract_segment sobre el WAV/FLAC original
# Uso: python 02_extract.py <raiz_proyecto> <dir_intermedios> [n_workers] [limite_por_sesion]
import os, sys, csv, glob, time
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool

ROOT, INT = sys.argv[1], sys.argv[2]
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 6
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
MIN_SEG = 2.0
os.makedirs(f'{INT}/feat', exist_ok=True)

def orig_path(sid):
    return f'{ROOT}/audio_s19/{sid}.flac' if sid.endswith('19') else f'{ROOT}/files_english/{sid}_FULL.wav'

def f0stats(snd, lo, hi):
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    f = p.selected_array['frequency']; f = f[f > 0]
    if len(f) < 10: return [np.nan] * 3 + [len(f)]
    st = 12 * np.log2(f / 100)
    return [float(np.median(f)), float(np.std(st)), float(np.percentile(st, 90) - np.percentile(st, 10)), len(f)]

def work(args):
    sid, rows = args
    out_csv = f'{INT}/feat/{sid}.csv'
    if os.path.exists(out_csv) and LIMIT is None: return sid, 0, 0.0
    import torch; torch.set_num_threads(1)
    import soundfile as sf, librosa, parselmouth
    sys.path.insert(0, f'{ROOT}/app_english'); import acoustic
    from pyannote.audio import Model, Inference
    inf = Inference(Model.from_pretrained('pyannote/wespeaker-voxceleb-resnet34-LM'), window='whole')
    t0 = time.time()
    y16, _ = librosa.load(f'{ROOT}/colab_upload/{sid}.mp3', sr=16000, mono=True)
    op = orig_path(sid); sr = sf.info(op).samplerate
    if LIMIT: rows = rows[:LIMIT]
    E, ids, recs = [], [], []
    for r in rows:
        a, b = float(r['seg_start']), float(r['seg_end'])
        # embedding
        x = y16[int(a * 16000):int(min(b, a + 12) * 16000)]
        e = np.asarray(inf({'waveform': torch.from_numpy(x.astype('float32'))[None, :], 'sample_rate': 16000})).ravel()
        # prosodia robusta sobre el original
        z, _ = sf.read(op, start=int(a * sr), stop=int(b * sr), dtype='float64', always_2d=True); z = z.mean(1)
        snd = parselmouth.Sound(z, sr)
        m = f0stats(snd, 60, 300); f = f0stats(snd, 120, 500)
        iv = snd.to_intensity(minimum_pitch=100).values[0]; iv = iv[iv > 0]
        pr = dict(m_f0med=m[0], m_f0sd_st=m[1], m_span_st=m[2], m_voiced=m[3],
                  f_f0med=f[0], f_f0sd_st=f[1], f_span_st=f[2], f_voiced=f[3],
                  int_med=float(np.median(iv)), int_sd=float(np.std(iv)), int_p90=float(np.percentile(iv, 90)))
        # rasgos AcusTera
        ac = acoustic.extract_segment(op, a, b)
        recs.append({'utt_id': r['utt_id'], **pr, **ac}); E.append(e); ids.append(r['utt_id'])
    if LIMIT: return sid, len(recs), time.time() - t0
    with open(out_csv + '.tmp', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    np.savez(f'{INT}/feat/{sid}_emb.npz', ids=np.array(ids), E=np.stack(E))
    os.replace(out_csv + '.tmp', out_csv)
    return sid, len(recs), time.time() - t0

if __name__ == '__main__':
    U = [r for r in csv.DictReader(open(f'{INT}/utterances.csv')) if float(r['duration']) >= MIN_SEG]
    by = {}
    for r in U: by.setdefault(f"{r['couple']}_{r['therapy']}", []).append(r)
    jobs = sorted(by.items(), key=lambda kv: -len(kv[1]))
    if LIMIT: jobs = jobs[:NW]
    with Pool(NW) as p:
        for sid, n, dt in p.imap_unordered(work, jobs):
            print(f'{sid:22} n={n:5} {dt:7.1f}s', flush=True)
