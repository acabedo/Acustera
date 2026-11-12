# Paso 3: asignación de roles por pareja (método validado de analisis_prosodico_2026-11:
# roles2 -> clasificador de sexo de voz; roles3 -> KMeans k=3 por pareja con embeddings centrados
# por sesión + pistas textuales del terapeuta; roles_final -> reasignación por centroides con
# restricciones de sexo, margen < 0,05 y alucinaciones).
# Uso: python 03_roles.py <dir_intermedios> <features_antiguas.csv> <roles_final_antiguo.csv>
# Solo imprime agregados.
import csv, re, sys, glob, collections, numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

INT, OLDF, OLDR = sys.argv[1:4]
U = {r['utt_id']: r for r in csv.DictReader(open(f'{INT}/utterances.csv'))}
P, ids, E = {}, [], []
for f in sorted(glob.glob(f'{INT}/feat/*_emb.npz')):
    z = np.load(f); ids += list(z['ids']); E.append(z['E'])
    for r in csv.DictReader(open(f.replace('_emb.npz', '.csv'))): P[r['utt_id']] = r
E0 = np.vstack(E); E0 = E0 / np.linalg.norm(E0, axis=1, keepdims=True)
ids = np.array(ids); N = len(ids)
fl = lambda x: float(x) if x not in (None, '', 'nan', 'NA') else np.nan
f_m = np.array([fl(P[i]['m_f0med']) for i in ids]); f_f = np.array([fl(P[i]['f_f0med']) for i in ids])
cpl = np.array([U[i]['couple'] for i in ids]); ses = np.array([int(U[i]['therapy_num']) for i in ids])
print('utterances with embeddings', N)

# --- (a) clasificador de sexo de voz (roles2)
male_seed = (f_m < 130) & (f_f < 160); fem_seed = (f_f > 195) & (f_m > 180)
X = np.r_[E0[male_seed], E0[fem_seed]]; y = np.r_[np.ones(male_seed.sum()), np.zeros(fem_seed.sum())]
clf = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')
print('sex seeds male/fem', male_seed.sum(), fem_seed.sum(), 'CV acc %.3f' % cross_val_score(clf, X, y, cv=5).mean())
pm = clf.fit(X, y).predict_proba(E0)[:, 1]

# --- pistas textuales (2.ª pl., discurso terapéutico, voseo)
VOS = re.compile(r'\b(vosotros|vosotras|est[aá]is|hab[eé]is|ten[eé]is|pod[eé]is|quer[eé]is|sent[ií]s|hac[eé]is|dec[ií]s|os|vuestr[oa]s?|acord[aá]is)\b')
TH = re.compile(r'(para ti\b|te sientes|qu[eé] sientes|c[oó]mo te|c[oó]mo es para|me pregunto|como que t[uú]|lo que necesitas|te escucho|te entiendo|un poquito)')
VOSEO = re.compile(r'\b(contame|contadme|decime|mir[aá]|fijate|escuchame|ten[eé]s|quer[eé]s|pod[eé]s|sab[eé]s|pens[aá]s|hac[eé]s|sos|vos)\b')
txt = lambda i: (U[i]['text'] or '').lower()
def thscore(ix):
    t = ' '.join(txt(ids[k]) for k in ix); nw = max(1, len(t.split()))
    return 1000 * (len(VOS.findall(t)) + len(TH.findall(t)) + len(VOSEO.findall(t))) / nw
def halluc(t):
    w = re.findall(r'\w+', (t or '').lower())
    return len(w) >= 8 and len(set(w)) / len(w) < 0.35

# --- (b) k=3 por pareja (roles3) + (c) reasignación final (roles_final)
role = np.empty(N, dtype=object); flag = np.empty(N, dtype=object); margin = np.zeros(N)
diag = []; th_cent = {}
for c in sorted(set(cpl)):
    idx = np.where(cpl == c)[0]; s = ses[idx]
    Ec = E0[idx].copy()
    for v in set(s): Ec[s == v] -= Ec[s == v].mean(0)
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    km = min((KMeans(3, n_init=10, random_state=r).fit(Ec) for r in range(5)), key=lambda m: m.inertia_)
    lab = km.labels_
    sc = [thscore(idx[lab == j]) for j in range(3)]; th = int(np.argmax(sc))
    rest = [j for j in range(3) if j != th]
    med = [np.nanmedian(f_m[idx[lab == j]]) for j in rest]
    man = rest[int(np.argmin(med))]; wom = rest[1 - int(np.argmin(med))]
    name = {th: 'therapist', man: 'man', wom: 'woman'}
    lab3 = np.array([name[l] for l in lab])
    roles = ['man', 'woman', 'therapist']
    cen = {r: Ec[lab3 == r].mean(0) for r in roles}; cen = {r: v / np.linalg.norm(v) for r, v in cen.items()}
    th_male = pm[idx[lab3 == 'therapist']].mean() > 0.5
    for j, k in enumerate(idx):
        sims = {r: float(Ec[j] @ cen[r]) for r in roles}
        allowed = roles
        if pm[k] > 0.7: allowed = ['man', 'therapist'] if th_male else ['man']
        elif pm[k] < 0.3: allowed = ['woman'] if th_male else ['woman', 'therapist']
        best = max(allowed, key=lambda r: sims[r])
        mg = sims[best] - max(sims[r] for r in roles if r != best)
        role[k] = best; margin[k] = mg
        flag[k] = 'hallucination' if halluc(U[ids[k]]['text']) else ('ambiguous' if mg < 0.05 else 'ok')
    ok = idx[flag[idx] == 'ok']
    pmc = {r: pm[idx[lab3 == r]].mean() for r in roles}
    f0c = {r: np.nanmedian(np.where(role[ok] == r, np.where(r == 'man' or (r == 'therapist' and th_male), f_m[ok], f_f[ok]), np.nan)) for r in roles}
    th_cent[c] = E0[ok[role[ok] == 'therapist']].mean(0)
    diag.append((c, 'th-score m/w/t', '/'.join('%.1f' % sc[j] for j in (man, wom, th)),
                 'pmale m/w/t', '/'.join('%.2f' % pmc[r] for r in roles),
                 'F0 m/w/t', '/'.join('%.0f' % f0c[r] for r in roles), 'th_male', th_male,
                 'changed vs k3 %.2f' % np.mean(role[idx] != lab3)))
for d in diag: print(*d)
cnt = collections.Counter(flag); print('flags', dict(cnt))

# --- comprobación 1: identidad de la terapeuta (similitud coseno entre centroides sin centrar)
cs = sorted(th_cent); M = np.array([th_cent[c] / np.linalg.norm(th_cent[c]) for c in cs]) ; S = M @ M.T
print('therapist centroid cosine similarity'); print('          ' + ' '.join(f'{c[:9]:>9}' for c in cs))
for i, c in enumerate(cs): print(f'{c:10}' + ' '.join(f'{S[i, j]:9.2f}' for j in range(len(cs))))

# --- comprobación 2: concordancia temporal con los roles validados del análisis muestreado
OF = {r['row_id']: r for r in csv.DictReader(open(OLDF))}
OR = [r for r in csv.DictReader(open(OLDR)) if r['flag'] == 'ok']
newby = collections.defaultdict(list)
for k in np.where(flag == 'ok')[0]:
    u = U[ids[k]]; newby[(u['couple'], int(u['therapy_num']))].append((float(u['seg_start']), float(u['seg_end']), role[k]))
agree = collections.Counter(); tot = collections.Counter(); conf = collections.Counter()
for r in OR:
    o = OF[r['row_id']]; key = (o['couple'], int(o['therapy_num'])); a, b = float(o['seg_start']), float(o['seg_end'])
    w = collections.Counter()
    for s, e, rl in newby[key]:
        ov = min(b, e) - max(a, s)
        if ov > 0: w[rl] += ov
    if not w: continue
    nr = w.most_common(1)[0][0]
    tot[o['couple']] += 1; agree[o['couple']] += (nr == r['role']); conf[(r['role'], nr)] += 1
print('agreement with validated sampled-mode roles (time overlap):',
      ' '.join(f'{c}={agree[c] / tot[c]:.2f}(n={tot[c]})' for c in sorted(tot)),
      '| overall %.3f' % (sum(agree.values()) / sum(tot.values())))
print('confusion old->new', sorted(conf.items()))

# --- comprobación 3: pureza de cada hablante DiariZen (proporción de duración del rol mayoritario)
dur = collections.defaultdict(collections.Counter)
for k in np.where(flag == 'ok')[0]:
    u = U[ids[k]]; dur[(u['couple'], u['therapy'], u['diar_spk'])][role[k]] += float(u['duration'])
pur = {kk: (v.most_common(1)[0][0], v.most_common(1)[0][1] / sum(v.values()), sum(v.values())) for kk, v in dur.items()}
pv = np.array([p[1] for p in pur.values()]); wv = np.array([p[2] for p in pur.values()])
print('DiariZen speaker purity: weighted mean %.3f | speakers with purity < .8: %d of %d' % ((pv * wv).sum() / wv.sum(), (pv < .8).sum(), len(pv)))
for kk, p in sorted(pur.items()):
    if p[1] < .8: print('   low purity', kk, p[0], '%.2f' % p[1], '%.0fs' % p[2])

# --- por sesión: recuento de enunciados ok por rol
S2 = collections.defaultdict(collections.Counter)
for k in np.where(flag == 'ok')[0]: S2[(cpl[k], ses[k])][role[k]] += 1
for kk in sorted(S2): print('  ', kk, dict(S2[kk]))

with open(f'{INT}/roles_final.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['utt_id', 'role', 'flag', 'margin', 'p_male'])
    for k in range(N): w.writerow([ids[k], role[k], flag[k], '%.3f' % margin[k], '%.3f' % pm[k]])
print('saved roles_final.csv')
