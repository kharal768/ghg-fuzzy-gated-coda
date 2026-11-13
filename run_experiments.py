"""
run_experiments.py -- regenerates every number reported in the revised manuscript.

Usage:  python run_experiments.py <data_csv> <out_dir>

Outputs (JSON/CSV) in <out_dir>:
  data_audit.json            composition construction, closure residuals, zero structure
  clustering_<cfg>.json      FCM validity indices C=2..10, gas correspondence, zero-pattern baseline,
                             zero-replacement sensitivity
  cv_raw_<cfg>.csv           test MSE for every repeat x fold x model x seed
  cv_summary_<cfg>.json      split-level means, SDs, corrected repeated-CV t-tests (Holm)
  gates_<cfg>.json           learned-gate selectivity and stability analysis
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghg_core as G

DELTA = 1e-5          # primary zero-replacement value (share)
N_REPEAT, N_FOLD, N_SEED = 5, 5, 3
M_FUZZ = 2.0

CONFIGS = {
    # all seven gases, 20 non-overlapping CRF parts (the object of the original submission)
    'all20': dict(gases=None, scheme=None, H=64, L=4, K=4),
    # multi-source gases only, 8-part amalgamation (substantive within-gas variation)
    'multi8': dict(gases=['CO2', 'CH4', 'N2O'], scheme=G.AMALG_8, H=32, L=2, K=3),
}


def build(cfg, path, delta=DELTA):
    c = CONFIGS[cfg.split('_L')[0]]
    d = G.load_inventory(path, c['gases'])
    if c['scheme'] is None:
        X, parts = d[G.PARTS_20].to_numpy(float), list(G.PARTS_20)
    else:
        X, parts = G.amalgamate(d, c['scheme'])
    Xr = G.mult_replace(X, delta)
    V = G.ilr_basis(Xr.shape[1])
    Z = G.ilr(Xr, V)
    return d, X, Xr, Z, V, parts


# ----------------------------------------------------------------------------------------------
def data_audit(path, out):
    d = G.load_inventory(path)
    X = d[G.PARTS_20].to_numpy(float)
    res = (d.TOT.to_numpy() - X.sum(1)) / d.TOT.to_numpy()
    zero = (X == 0)
    audit = dict(
        n_rows=int(len(d)), n_countries=int(d.geo.nunique()),
        countries=sorted(d.geo.unique().tolist()),
        rows_per_gas=d.gas.value_counts().reindex(G.GAS_ORDER).fillna(0).astype(int).to_dict(),
        n_parts=len(G.PARTS_20), parts=G.PARTS_20,
        negative_cells=int((X < 0).sum()),
        zero_fraction_all=float(zero.mean()),
        zero_fraction_by_gas={g: float(zero[d.gas.values == g].mean()) for g in G.GAS_ORDER},
        rows_with_single_nonzero_part=int((~zero).sum(1).__eq__(1).sum()),
        nonzero_parts_per_row_by_gas={g: dict(zip(['min', 'median', 'max'],
                                      np.percentile((~zero)[d.gas.values == g].sum(1), [0, 50, 100]).tolist()))
                                      for g in G.GAS_ORDER},
        closure_residual_abs_max=float(np.abs(res).max()),
        closure_residual_abs_median=float(np.median(np.abs(res))),
        closure_rows_abs_resid_gt_1pct=int((np.abs(res) > 0.01).sum()),
        closure_worst_row=d.iloc[int(np.abs(res).argmax())][['geo', 'gas']].tolist(),
    )
    for cfg in CONFIGS:
        dd, Xc, Xr, Z, V, parts = build(cfg, path)
        audit[f'{cfg}_n'] = int(len(dd)); audit[f'{cfg}_D'] = len(parts)
        audit[f'{cfg}_zero_fraction'] = float((Xc == 0).mean())
        audit[f'{cfg}_ilr_roundtrip_maxerr'] = float(np.abs(G.ilr_inv(Z, V) - Xr).max())
        audit[f'{cfg}_parts'] = parts
    audit['grad_check_max_rel_err'] = {g: G.grad_check(g) for g in ['none', 'concat', 'side', 'moe', 'nf']}
    json.dump(audit, open(os.path.join(out, 'data_audit.json'), 'w'), indent=1)
    return audit


# ----------------------------------------------------------------------------------------------
def clustering(cfg, path, out):
    d, X, Xr, Z, V, parts = build(cfg, path)
    gas = d.gas.to_numpy()
    rows = []
    fits = {}
    for C in range(2, 11):
        cen, U, J = G.fcm(Z, C, m=M_FUZZ, n_init=15, seed=C)
        ind = G.fcm_indices(Z, U, cen, M_FUZZ)
        ind['J'] = J
        ind['ARI_gas'] = float(adjusted_rand_score(gas, U.argmax(1)))
        km = KMeans(C, n_init=20, random_state=0).fit(Z)
        ind['ARI_gas_kmeans'] = float(adjusted_rand_score(gas, km.labels_))
        rows.append(ind); fits[C] = (cen, U)
    tab = pd.DataFrame(rows)
    # choice of C: maximise silhouette (ties -> smaller C); reported alongside all indices
    C_star = int(tab.sort_values(['SS', 'C'], ascending=[False, True]).iloc[0].C)
    cen, U = fits[C_star]
    lab = U.argmax(1)
    ct = pd.crosstab(pd.Series(lab, name='cluster'), pd.Series(gas, name='gas'))
    ct = ct.reindex(columns=[g for g in G.GAS_ORDER if g in ct.columns])
    cen_simplex = G.ilr_inv(cen, V)
    top = {}
    for k in range(C_star):
        o = np.argsort(-cen_simplex[k])[:3]
        top[int(k)] = [[parts[j], float(cen_simplex[k, j])] for j in o]
    # zero-pattern-only baseline: cluster the binary support pattern (Jaccard, average linkage)
    B = (X > 0)
    Dj = squareform(pdist(B, metric='jaccard'))
    zp = AgglomerativeClustering(n_clusters=C_star, metric='precomputed', linkage='average').fit(Dj).labels_
    ari_zero_pattern = float(adjusted_rand_score(gas, zp))
    ari_fcm_vs_zero = float(adjusted_rand_score(lab, zp))
    # zero-replacement sensitivity
    sens = []
    for dl in [1e-6, 1e-5, 1e-4, 1e-3]:
        _, _, _, Zs, _, _ = build(cfg, path, delta=dl)
        cs, Us, _ = G.fcm(Zs, C_star, m=M_FUZZ, n_init=15, seed=C_star)
        ii = G.fcm_indices(Zs, Us, cs, M_FUZZ)
        ii['delta'] = dl
        ii['ARI_gas'] = float(adjusted_rand_score(gas, Us.argmax(1)))
        ii['ARI_vs_primary'] = float(adjusted_rand_score(lab, Us.argmax(1)))
        ii['C_star_by_SS'] = None
        tt = []
        for C in range(2, 11):
            c2, U2, _ = G.fcm(Zs, C, m=M_FUZZ, n_init=10, seed=C)
            tt.append((G.fcm_indices(Zs, U2, c2, M_FUZZ)['SS'], C))
        ii['C_star_by_SS'] = int(sorted(tt, key=lambda t: (-t[0], t[1]))[0][1])
        sens.append(ii)
    res = dict(cfg=cfg, n=int(len(Z)), D=len(parts), delta=DELTA, m=M_FUZZ,
               indices=rows, C_star=C_star, crosstab=ct.to_dict(orient='index'),
               cluster_sizes=np.bincount(lab, minlength=C_star).tolist(),
               centroid_top_parts=top, ARI_fcm_gas=float(adjusted_rand_score(gas, lab)),
               ARI_zero_pattern_gas=ari_zero_pattern, ARI_fcm_vs_zero_pattern=ari_fcm_vs_zero,
               delta_sensitivity=sens,
               max_membership_summary=dict(zip(['min', 'q25', 'median', 'q75', 'max'],
                                           np.percentile(U.max(1), [0, 25, 50, 75, 100]).tolist())))
    json.dump(res, open(os.path.join(out, f'clustering_{cfg}.json'), 'w'), indent=1)
    np.savez(os.path.join(out, f'fcm_full_{cfg}.npz'), Z=Z, U=U, cen=cen, gas=gas, geo=d.geo.to_numpy())
    return res


# ----------------------------------------------------------------------------------------------
def country_folds(geos, rep):
    u = np.array(sorted(set(geos)))
    r = np.random.default_rng(1000 + rep)
    r.shuffle(u)
    groups = np.array_split(u, N_FOLD)
    return [np.isin(geos, g) for g in groups]


def one_split(args):
    cfg, path, rep, fold, C_star = args
    c = CONFIGS[cfg]
    d, X, Xr, Z, V, parts = build(cfg, path)
    geos, gas = d.geo.to_numpy(), d.gas.to_numpy()
    te = country_folds(geos, rep)[fold]
    tr_all = ~te
    # inner validation: 20% of training countries (for early stopping only)
    tr_c = np.array(sorted(set(geos[tr_all])))
    r = np.random.default_rng(5000 + 10 * rep + fold)
    va_c = r.choice(tr_c, size=max(1, int(round(0.2 * len(tr_c)))), replace=False)
    va = tr_all & np.isin(geos, va_c)
    tr = tr_all & ~va
    mu = Z[tr].mean(0)                         # centring = perturbation by training centre
    Ztr, Zva, Zte = Z[tr] - mu, Z[va] - mu, Z[te] - mu
    # FCM on training rows only; memberships for val/test from frozen centres
    cen, Utr, _ = G.fcm(Ztr, C_star, m=M_FUZZ, n_init=15, seed=rep * 100 + fold)
    Uva, Ute = G.fcm_predict(Zva, cen, M_FUZZ), G.fcm_predict(Zte, cen, M_FUZZ)
    hard = lambda U: np.eye(C_star)[U.argmax(1)]
    gl = sorted(set(gas))
    onehot = lambda m: np.eye(len(gl))[[gl.index(x) for x in gas[m]]]
    pr = np.random.default_rng(777 + rep * 10 + fold)
    perm = lambda U: U[pr.permutation(len(U))]
    side = {
        'AE-concat': ('concat', Utr, Uva, Ute),
        'AE-gate-FCM': ('side', Utr, Uva, Ute),
        'AE-gate-hard': ('side', hard(Utr), hard(Uva), hard(Ute)),
        'AE-gate-random': ('side', perm(Utr), perm(Uva), perm(Ute)),
        'AE-gate-gaslabel': ('side', onehot(tr), onehot(va), onehot(te)),
    }
    dz = Z.shape[1]
    H, L, K = c['H'], c['L'], c['K']
    n_gate = AE_params = G.AE(dz, H, L, 'side', s_dim=C_star).n_params()
    H_match = G.matched_hidden(dz, L, n_gate)
    out, gate_rec = [], []
    for seed in range(N_SEED):
        models = {
            'AE-plain': (G.AE(dz, H, L, 'none', seed=seed), None, None, None),
            'AE-plain-matched': (G.AE(dz, H_match, L, 'none', seed=seed), None, None, None),
            'AE-MoE-learned': (G.AE(dz, H, L, 'moe', K=K, seed=seed), None, None, None),
            'AE-neurofuzzy': (G.AE(dz, H, L, 'nf', K=K, seed=seed), None, None, None),
        }
        for name, (gt, a, b, cc) in side.items():
            models[name] = (G.AE(dz, H, L, gt, s_dim=a.shape[1], seed=seed), a, b, cc)
        for name, (net, a, b, cc) in models.items():
            t0 = time.time()
            net.fit(Ztr, a, Zva, b, seed=seed)
            mse = net.loss(Zte, cc)
            out.append(dict(cfg=cfg, rep=rep, fold=fold, seed=seed, model=name, mse=mse,
                            n_params=net.n_params(), epoch=net.best_epoch, sec=time.time() - t0,
                            n_tr=int(tr.sum()), n_va=int(va.sum()), n_te=int(te.sum())))
            if name == 'AE-gate-FCM':
                gtr = net.gates(Ztr, a)
                lab = a.argmax(1)
                prof = np.array([gtr[lab == k].mean(0) if (lab == k).any() else np.full(H, np.nan)
                                 for k in range(C_star)])
                gate_rec.append(dict(rep=rep, fold=fold, seed=seed, profiles=prof.tolist(),
                                     gate_mean=float(gtr.mean()), gate_sd=float(gtr.std()),
                                     train_gas=gas[tr].tolist(), train_lab=lab.tolist()))
        # classical baselines (seeded where stochastic)
        pca = PCA(L).fit(Ztr)
        out.append(dict(cfg=cfg, rep=rep, fold=fold, seed=seed, model='PCA', mse=float(
            ((pca.inverse_transform(pca.transform(Zte)) - Zte) ** 2).mean()), n_params=int(pca.components_.size + dz),
            epoch=0, sec=0, n_tr=int(tr.sum()), n_va=int(va.sum()), n_te=int(te.sum())))
        rf = RandomForestRegressor(300, random_state=seed, n_jobs=1).fit(pca.transform(np.vstack([Ztr, Zva])),
                                                                         np.vstack([Ztr, Zva]))
        out.append(dict(cfg=cfg, rep=rep, fold=fold, seed=seed, model='RF-on-PCA', mse=float(
            ((rf.predict(pca.transform(Zte)) - Zte) ** 2).mean()), n_params=0, epoch=0, sec=0,
            n_tr=int(tr.sum()), n_va=int(va.sum()), n_te=int(te.sum())))
    # mean-only baseline
    out.append(dict(cfg=cfg, rep=rep, fold=fold, seed=0, model='Mean', mse=float((Zte ** 2).mean()),
                    n_params=dz, epoch=0, sec=0, n_tr=int(tr.sum()), n_va=int(va.sum()), n_te=int(te.sum())))
    # FCM partition of the training fold, for stability analysis
    part = dict(rep=rep, fold=fold, train_idx=np.where(tr)[0].tolist(), lab=Utr.argmax(1).tolist())
    return out, gate_rec, part, dict(H_match=H_match, n_gate_params=n_gate)


def corrected_t(diff, n_tr, n_te):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated K-fold CV."""
    J = len(diff)
    m, v = diff.mean(), diff.var(ddof=1)
    t = m / np.sqrt((1.0 / J + n_te / n_tr) * v) if v > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), J - 1)
    return float(m), float(t), float(p)


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def cv(cfg, path, out, C_star, procs=2):
    jobs = [(cfg, path, r, f, C_star) for r in range(N_REPEAT) for f in range(N_FOLD)]
    with Pool(procs) as pool:
        res = pool.map(one_split, jobs)
    raw = pd.DataFrame([row for o, _, _, _ in res for row in o])
    raw.to_csv(os.path.join(out, f'cv_raw_{cfg}.csv'), index=False)
    gates = [g for _, gr, _, _ in res for g in gr]
    parts = [p for _, _, p, _ in res]
    meta = res[0][3]
    split = raw.groupby(['model', 'rep', 'fold']).mse.mean().unstack('model')
    n_tr = raw.n_tr.mean() + raw.n_va.mean(); n_te = raw.n_te.mean()
    summ = {}
    for mname in split.columns:
        summ[mname] = dict(mean=float(split[mname].mean()), sd_over_splits=float(split[mname].std()),
                           seed_sd_mean=float(raw[raw.model == mname].groupby(['rep', 'fold']).mse.std().mean()
                                              if mname != 'Mean' else 0.0),
                           n_params=int(raw[raw.model == mname].n_params.iloc[0]))
    tests = {}
    for ref in ['AE-plain', 'PCA']:
        names = [m for m in split.columns if m != ref and m != 'Mean']
        rr = [corrected_t((split[m] - split[ref]).to_numpy(), n_tr, n_te) for m in names]
        adj = holm([x[2] for x in rr])
        wil = [float(stats.wilcoxon(split[m] - split[ref]).pvalue) for m in names]
        tests[ref] = {m: dict(mean_diff=x[0], t=x[1], p=x[2], p_holm=float(a), p_wilcoxon_naive=w,
                              wins=int((split[m] < split[ref]).sum()), n_splits=int(len(split)))
                      for m, x, a, w in zip(names, rr, adj, wil)}
    json.dump(dict(cfg=cfg, C_star=C_star, n_splits=int(len(split)), n_seeds=N_SEED,
                   n_train_mean=float(raw.n_tr.mean()), n_val_mean=float(raw.n_va.mean()),
                   n_test_mean=float(raw.n_te.mean()), meta=meta, summary=summ, tests=tests),
              open(os.path.join(out, f'cv_summary_{cfg}.json'), 'w'), indent=1)
    # ---- gate analysis -----------------------------------------------------------------------
    sel, inactive, between = [], [], []
    for g in gates:
        P = np.array(g['profiles'])
        P = P[~np.isnan(P).any(1)]
        rng_k = P.max(0) - P.min(0)                       # per-unit between-cluster gate range
        sel.append(float((rng_k > 0.5).mean()))           # fraction of strongly cluster-selective units
        inactive.append(float((P.max(0) < 0.1).mean()))   # units shut for every cluster
        between.append(float(rng_k.mean()))
    # FCM partition stability across training folds: ARI on overlapping observations
    aris = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = parts[i], parts[j]
            common, ia, ib = np.intersect1d(a['train_idx'], b['train_idx'], return_indices=True)
            if len(common) > 20:
                aris.append(adjusted_rand_score(np.array(a['lab'])[ia], np.array(b['lab'])[ib]))
    # gate-profile reproducibility across seeds within a split: compare sorted between-cluster ranges
    gdf = pd.DataFrame(dict(rep=[g['rep'] for g in gates], fold=[g['fold'] for g in gates], sel=sel,
                            between=between, inactive=inactive))
    json.dump(dict(cfg=cfg, frac_units_selective_gt0p5=dict(mean=float(np.mean(sel)), sd=float(np.std(sel))),
                   frac_units_closed_all_clusters=dict(mean=float(np.mean(inactive)), sd=float(np.std(inactive))),
                   mean_between_cluster_gate_range=dict(mean=float(np.mean(between)), sd=float(np.std(between))),
                   selectivity_sd_across_seeds_within_split=float(gdf.groupby(['rep', 'fold']).sel.std().mean()),
                   fcm_partition_stability_ARI=dict(mean=float(np.mean(aris)), min=float(np.min(aris)),
                                                    max=float(np.max(aris)), n_pairs=len(aris))),
              open(os.path.join(out, f'gates_{cfg}.json'), 'w'), indent=1)
    # store one representative gate profile for the figure
    np.save(os.path.join(out, f'gate_profile_example_{cfg}.npy'), np.array(gates[0]['profiles']))
    return summ, tests


if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    print(json.dumps(data_audit(path, out), indent=1)[:1500])
    for cfg in CONFIGS:
        cl = clustering(cfg, path, out)
        print(cfg, 'C*', cl['C_star'], 'ARI', cl['ARI_fcm_gas'], 'ARI zero-pattern', cl['ARI_zero_pattern_gas'])
        summ, tests = cv(cfg, path, out, cl['C_star'])
        for k, v in sorted(summ.items(), key=lambda kv: kv[1]['mean']):
            print(f"  {k:20s} {v['mean']:.4f} +- {v['sd_over_splits']:.4f}")
        print('  elapsed', round(time.time() - t0), 's')
