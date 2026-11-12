# Paso 2: palabras CrisperWhisper + turnos DiariZen -> enunciados (lógica de AcusTera:
# palabras asignadas al turno por solapamiento, corte en pausas >= 0,3 s).
# Uso: python 01_utterances.py <dir_resultados> <dir_salida>
# Solo imprime agregados (no texto).
import json, glob, os, sys, re, csv, collections
import numpy as np

RES, OUT = sys.argv[1], sys.argv[2]
PAUSE = 0.3          # acustera.pause_split_s
SPUR_S = 60.0        # hablante DiariZen con < 60 s en la sesión = espurio
SNAP = 0.3           # palabra fuera de turno: se pega al turno más cercano si está a <= 0,3 s

def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def overlap_regions(turns):
    """Intervalos donde hablan >=2 hablantes distintos (barrido de eventos)."""
    ev = []
    for t in turns: ev += [(t['ini'], 1, t['hablante']), (t['fin'], -1, t['hablante'])]
    ev.sort(key=lambda e: (e[0], e[1]))
    act = collections.Counter(); reg = []; cur = None
    for x, d, s in ev:
        nspk = sum(1 for v in act.values() if v > 0)
        if nspk >= 2 and cur is not None and x > cur: reg.append([cur, x])
        act[s] += d; cur = x
    return merge(reg)

utt_rows, ov_rows, log = [], [], []
for fcw in sorted(glob.glob(f'{RES}/*_crisperwhisper.json')):
    sid = os.path.basename(fcw).replace('_crisperwhisper.json', '')
    couple, th = sid.split('_'); tnum = int(re.sub(r'\D', '', th))
    cw = json.load(open(fcw)); dz = json.load(open(f'{RES}/{sid}_diarizen.json'))
    words = [w for w in cw['palabras'] if w['palabra'].strip()]
    for w in words:
        if w['fin'] <= w['ini']: w['fin'] = w['ini'] + 0.02
    T = dz['turnos']
    tot = collections.Counter()
    for t in T: tot[t['hablante']] += t['fin'] - t['ini']
    spur = {s for s, d in tot.items() if d < SPUR_S}
    T = sorted([t for t in T if t['hablante'] not in spur], key=lambda t: t['ini'])
    starts = np.array([t['ini'] for t in T]); ends = np.array([t['fin'] for t in T])
    total_dur = max(cw.get('fin_s') or 0, max(w['fin'] for w in words), ends.max())
    # palabra -> turno
    assign = []; n_snap = n_drop = 0
    for w in words:
        a, b = w['ini'], w['fin']
        ov = np.minimum(ends, b) - np.maximum(starts, a)
        j = int(np.argmax(ov))
        if ov[j] > 0: assign.append(j); continue
        m = (a + b) / 2
        dist = np.maximum(starts - m, 0) + np.maximum(m - ends, 0)
        j = int(np.argmin(dist))
        if dist[j] <= SNAP: assign.append(j); n_snap += 1
        else: assign.append(-1); n_drop += 1
    by = collections.defaultdict(list)
    for w, j in zip(words, assign):
        if j >= 0: by[j].append(w)
    ovreg = overlap_regions(T)
    ovs = np.array([r[0] for r in ovreg]) if ovreg else np.zeros(0)
    ove = np.array([r[1] for r in ovreg]) if ovreg else np.zeros(0)
    k = 0
    for j in sorted(by):
        ws = sorted(by[j], key=lambda w: w['ini'])
        grp = [[ws[0]]]
        for p, q in zip(ws[:-1], ws[1:]):
            if q['ini'] - p['fin'] >= PAUSE: grp.append([q])
            else: grp[-1].append(q)
        for g in grp:
            s, e = g[0]['ini'], max(w['fin'] for w in g)
            ovd = float(np.clip(np.minimum(ove, e) - np.maximum(ovs, s), 0, None).sum()) if len(ovs) else 0.0
            mid = (s + e) / 2
            q = 'Q%d' % min(4, int(4 * mid / total_dur) + 1)
            utt_rows.append(dict(utt_id=f'{sid}_{k:05d}', couple=couple, therapy=th, therapy_num=tnum,
                                 diar_spk=T[j]['hablante'], turn_idx=j, seg_start=round(s, 3), seg_end=round(e, 3),
                                 duration=round(e - s, 3), n_words=len(g), is_overlap=ovd > 0,
                                 ov_frac=round(ovd / max(e - s, 1e-3), 3), quantile=q,
                                 text=' '.join(w['palabra'].strip() for w in g)))
            k += 1
    dur_ov = sum(b - a for a, b in ovreg)
    ov_rows.append(dict(couple=couple, therapy=th, therapy_num=tnum, dur_total_s=round(total_dur, 1),
                        n_overlappings=len(ovreg), dur_overlap_s=round(dur_ov, 2),
                        ratio_overlap=round(dur_ov / total_dur, 4)))
    log.append((sid, len(words), n_snap, n_drop, k, len(spur), len(tot) - len(spur), round(dur_ov / total_dur, 3)))

with open(f'{OUT}/utterances.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(utt_rows[0])); w.writeheader(); w.writerows(utt_rows)
with open(f'{OUT}/overlaps_diar.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(ov_rows[0])); w.writeheader(); w.writerows(ov_rows)
print('sesion words snapped dropped utts spk_spurious spk_kept ov_ratio')
for l in log: print(*l)
d = np.array([r['duration'] for r in utt_rows])
print('total utts', len(utt_rows), '| >=2s', int((d >= 2).sum()), '| median dur %.2f' % np.median(d),
      '| speech in >=2s utts %.0f%%' % (100 * d[d >= 2].sum() / d.sum()))
