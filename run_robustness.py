"""
run_robustness.py -- external cluster metrics, choice of C, fuzzifier sensitivity and confidence intervals.

Usage: python run_robustness.py <data_csv> <out_dir>

Writes to <out_dir>:
  robust_cluster_metrics_<cfg>.json  ARI / NMI / purity / fuzzy silhouette for FCM and k-means, C = 2..10
  robust_C_stability_<cfg>.json      subsampling stability of each C and of the silhouette-selected C
  robust_fuzzifier_<cfg>.json        sensitivity to the FCM fuzzifier m (clustering and gating CV)
  robust_ci.json                     corrected 95% CIs for the main CV comparisons
  robust_atypical_<cfg>.csv          rows with the weakest maximum FCM membership
"""
import sys, os, json
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_samples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghg_core as G
import run_experiments as R

N_SUB = 50          # country subsamples for C-stability
SUB_FRAC = 0.8
M_GRID = [1.5, 2.0, 2.5, 3.0]


def purity(y, lab):
    ct = pd.crosstab(lab, y)
    return float(ct.max(axis=1).sum() / len(y))


def fuzzy_silhouette(Z, U, alpha=1.0):
    """Campello & Hruschka (2006) fuzzy silhouette: crisp silhouettes weighted by the gap
    between the two largest memberships of each object."""
    lab = U.argmax(1)
    if len(np.unique(lab)) < 2:
        return np.nan
    s = silhouette_samples(Z, lab)
    srt = np.sort(U, axis=1)
    w = (srt[:, -1] - srt[:, -2]) ** alpha
    return float((w * s).sum() / w.sum())


def cluster_metrics(cfg, path, out):
    d, X, Xr, Z, V, parts = R.build(cfg, path)
    gas = d.gas.to_numpy()
    rows = []
    for C in range(2, 11):
        cen, U, _ = G.fcm(Z, C, m=2.0, n_init=15, seed=C)
        lab = U.argmax(1)
        km = KMeans(C, n_init=20, random_state=0).fit(Z).labels_
        rows.append(dict(C=C, FCM_ARI=float(ARI(gas, lab)), FCM_NMI=float(NMI(gas, lab)),
                         FCM_purity=purity(gas, lab), FCM_fuzzy_silhouette=fuzzy_silhouette(Z, U),
                         KM_ARI=float(ARI(gas, km)), KM_NMI=float(NMI(gas, km)), KM_purity=purity(gas, km)))
    json.dump(dict(cfg=cfg, rows=rows), open(os.path.join(out, f'robust_cluster_metrics_{cfg}.json'), 'w'), indent=1)
    return rows


def C_stability(cfg, path, out):
    """For each C: fit FCM on the full data and on N_SUB random 80% country subsamples; report the
    mean ARI between subsample and full-data partitions on the subsample rows, and how often each C
    is selected by the silhouette in the subsamples."""
    d, X, Xr, Z, V, parts = R.build(cfg, path)
    geos = d.geo.to_numpy()
    countries = np.array(sorted(set(geos)))
    full = {C: G.fcm(Z, C, m=2.0, n_init=15, seed=C)[1].argmax(1) for C in range(2, 11)}
    rng = np.random.default_rng(2026)
    stab = {C: [] for C in range(2, 11)}
    chosen = []
    for b in range(N_SUB):
        keep = np.isin(geos, rng.choice(countries, int(round(SUB_FRAC * len(countries))), replace=False))
        Zs = Z[keep]
        best = None
        for C in range(2, 11):
            cen, U, _ = G.fcm(Zs, C, m=2.0, n_init=8, seed=1000 + b * 10 + C)
            lab = U.argmax(1)
            stab[C].append(ARI(full[C][keep], lab))
            ss = G.fcm_indices(Zs, U, cen, 2.0)['SS']
            if best is None or ss > best[0] + 1e-12:
                best = (ss, C)
        chosen.append(best[1])
    res = dict(cfg=cfg, n_subsamples=N_SUB, frac_countries=SUB_FRAC,
               stability_ARI={str(C): dict(mean=float(np.mean(v)), sd=float(np.std(v)), q05=float(np.percentile(v, 5)))
                              for C, v in stab.items()},
               selected_C_frequency={str(c): int((np.array(chosen) == c).sum()) for c in range(2, 11)})
    json.dump(res, open(os.path.join(out, f'robust_C_stability_{cfg}.json'), 'w'), indent=1)
    return res


def _cv_split_m(args):
    cfg, path, rep, fold, C, m = args
    c = R.CONFIGS[cfg]
    d, X, Xr, Z, V, parts = R.build(cfg, path)
    geos = d.geo.to_numpy()
    te = R.country_folds(geos, rep)[fold]
    tr_c = np.array(sorted(set(geos[~te])))
    r = np.random.default_rng(5000 + 10 * rep + fold)
    va_c = r.choice(tr_c, size=max(1, int(round(0.2 * len(tr_c)))), replace=False)
    va = ~te & np.isin(geos, va_c); tr = ~te & ~va
    mu = Z[tr].mean(0); Ztr, Zva, Zte = Z[tr] - mu, Z[va] - mu, Z[te] - mu
    cen, Utr, _ = G.fcm(Ztr, C, m=m, n_init=15, seed=rep * 100 + fold)
    Uva, Ute = G.fcm_predict(Zva, cen, m), G.fcm_predict(Zte, cen, m)
    pr = np.random.default_rng(777 + rep * 10 + fold)
    perm = lambda U: U[pr.permutation(len(U))]
    P = (perm(Utr), perm(Uva), perm(Ute))
    out = []
    for seed in range(R.N_SEED):
        for name, gate, S in [('AE-plain', 'none', (None, None, None)),
                              ('AE-gate-FCM', 'side', (Utr, Uva, Ute)),
                              ('AE-gate-random', 'side', P)]:
            net = G.AE(Z.shape[1], c['H'], c['L'], gate, s_dim=C, seed=seed)
            net.fit(Ztr, S[0], Zva, S[1], seed=seed)
            out.append(dict(m=m, rep=rep, fold=fold, seed=seed, model=name, mse=net.loss(Zte, S[2]),
                            n_tr=int(tr.sum() + va.sum()), n_te=int(te.sum())))
    return out


def corrected(diff, n_tr, n_te):
    J = len(diff); mdiff = diff.mean(); se = np.sqrt((1.0 / J + n_te / n_tr) * diff.var(ddof=1))
    t = mdiff / se; p = 2 * stats.t.sf(abs(t), J - 1); q = stats.t.ppf(0.975, J - 1)
    return dict(mean_diff=float(mdiff), ci95=[float(mdiff - q * se), float(mdiff + q * se)], t=float(t), p=float(p))


def fuzzifier(cfg, path, out, C, procs=2):
    d, X, Xr, Z, V, parts = R.build(cfg, path)
    gas = d.gas.to_numpy()
    clus = []
    for m in M_GRID:
        cen, U, _ = G.fcm(Z, C, m=m, n_init=15, seed=C)
        ind = G.fcm_indices(Z, U, cen, m)
        lab = U.argmax(1)
        ind.update(m=m, ARI_gas=float(ARI(gas, lab)), NMI_gas=float(NMI(gas, lab)),
                   fuzzy_silhouette=fuzzy_silhouette(Z, U), median_max_membership=float(np.median(U.max(1))))
        tt = []
        for CC in range(2, 11):
            c2, U2, _ = G.fcm(Z, CC, m=m, n_init=10, seed=CC)
            tt.append((G.fcm_indices(Z, U2, c2, m)['SS'], CC))
        ind['C_selected_by_SS'] = int(sorted(tt, key=lambda t: (-t[0], t[1]))[0][1])
        clus.append(ind)
    jobs = [(cfg, path, r, f, C, m) for m in M_GRID for r in range(R.N_REPEAT) for f in range(R.N_FOLD)]
    with Pool(procs) as pool:
        raw = pd.DataFrame([x for o in pool.map(_cv_split_m, jobs) for x in o])
    raw.to_csv(os.path.join(out, f'robust_fuzzifier_cv_raw_{cfg}.csv'), index=False)
    cv = []
    for m in M_GRID:
        r = raw[raw.m == m]
        sp = r.groupby(['model', 'rep', 'fold']).mse.mean().unstack('model')
        n_tr, n_te = r.n_tr.mean(), r.n_te.mean()
        cv.append(dict(m=m, mse={k: float(sp[k].mean()) for k in sp.columns},
                       gateFCM_vs_plain=corrected((sp['AE-gate-FCM'] - sp['AE-plain']).to_numpy(), n_tr, n_te),
                       gateFCM_vs_random=corrected((sp['AE-gate-FCM'] - sp['AE-gate-random']).to_numpy(), n_tr, n_te)))
    json.dump(dict(cfg=cfg, C=C, clustering=clus, cv=cv), open(os.path.join(out, f'robust_fuzzifier_{cfg}.json'), 'w'), indent=1)
    return clus, cv


def main_cis(out):
    res = {}
    for cfg in ['all20', 'multi8', 'all20_L2', 'all20_L8']:
        raw = pd.read_csv(os.path.join(out, f'cv_raw_{cfg}.csv'))
        sp = raw.groupby(['model', 'rep', 'fold']).mse.mean().unstack('model')
        n_tr = raw.n_tr.mean() + raw.n_va.mean(); n_te = raw.n_te.mean()
        res[cfg] = {ref: {mm: corrected((sp[mm] - sp[ref]).to_numpy(), n_tr, n_te)
                          for mm in sp.columns if mm not in (ref, 'Mean')} for ref in ['AE-plain', 'PCA']}
    json.dump(res, open(os.path.join(out, 'robust_ci.json'), 'w'), indent=1)
    return res


def atypical(cfg, path, out):
    d, X, Xr, Z, V, parts = R.build(cfg, path)
    f = np.load(os.path.join(out, f'fcm_full_{cfg}.npz'), allow_pickle=True)
    U = f['U']
    srt = np.sort(U, axis=1)
    tab = pd.DataFrame(dict(geo=d.geo, gas=d.gas, max_membership=srt[:, -1], second=srt[:, -2],
                            cluster=U.argmax(1), second_cluster=np.argsort(U, axis=1)[:, -2]))
    tab = tab.sort_values('max_membership').reset_index(drop=True)
    tab.to_csv(os.path.join(out, f'robust_atypical_{cfg}.csv'), index=False)
    return tab


if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    print(json.dumps(main_cis(out)['all20']['AE-plain']['AE-gate-FCM']))
    for cfg in ['all20', 'multi8']:
        C = json.load(open(os.path.join(out, f'clustering_{cfg}.json')))['C_star']
        rows = cluster_metrics(cfg, path, out)
        print(cfg, [r for r in rows if r['C'] == C])
        st = C_stability(cfg, path, out)
        print(cfg, {k: round(v['mean'], 3) for k, v in st['stability_ARI'].items()}, st['selected_C_frequency'])
        clus, cv = fuzzifier(cfg, path, out, C)
        for a, b in zip(clus, cv):
            print(cfg, 'm', a['m'], 'Csel', a['C_selected_by_SS'], 'ARI', round(a['ARI_gas'], 3), 'medmax',
                  round(a['median_max_membership'], 3), 'gate-plain', round(b['gateFCM_vs_plain']['mean_diff'], 4),
                  'p', round(b['gateFCM_vs_plain']['p'], 3), 'vsRandom p', round(b['gateFCM_vs_random']['p'], 3))
        print(atypical(cfg, path, out).head(8).to_string())
