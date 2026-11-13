"""
run_ablation.py -- constant-gate ablation, non-parametric tests, Ridge/PCA check and within-gas FCM.

Usage: python run_ablation.py <data_csv> <out_dir>

  (a) Constant-gate ablation: the gate network is kept (same W_g, b_g, same parameter count) but its
      input is the uniform vector (1/C,...,1/C) for every row, i.e. the fuzzy clustering is removed
      and the gate reduces to a learned per-unit rescaling. Run on exactly the same 5x5 grouped
      splits and leave-one-country-out folds, seeds and early-stopping sets as the main analyses.
  (b) Non-parametric robustness: Wilcoxon signed-rank tests on the 25 split-level and the 31
      per-country paired differences; Shapiro-Wilk on the differences (t-test assumption check).
  (c) Ridge decoder on PCA scores versus PCA reconstruction (numerical equivalence check).
  (d) Within-gas FCM for CO2, CH4 and N2O separately (8-part data, n = 31 each).
Writes ablation_results.json.
"""
import sys, os, json
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghg_core as G
import run_experiments as R
import run_loco_bootstrap as LB


def const_cv(args):
    cfg, path, rep, fold, C = args
    c = R.CONFIGS[cfg]
    d, X, Xr, Z, V, parts = R.build(cfg, path)
    geos = d.geo.to_numpy()
    te = R.country_folds(geos, rep)[fold]
    tr_c = np.array(sorted(set(geos[~te])))
    r = np.random.default_rng(5000 + 10 * rep + fold)
    va_c = r.choice(tr_c, size=max(1, int(round(0.2 * len(tr_c)))), replace=False)
    va = ~te & np.isin(geos, va_c); tr = ~te & ~va
    mu = Z[tr].mean(0); Ztr, Zva, Zte = Z[tr] - mu, Z[va] - mu, Z[te] - mu
    k = lambda n: np.full((n, C), 1.0 / C)
    out = []
    for seed in range(R.N_SEED):
        net = G.AE(Z.shape[1], c['H'], c['L'], 'side', s_dim=C, seed=seed)
        net.fit(Ztr, k(len(Ztr)), Zva, k(len(Zva)), seed=seed)
        out.append(dict(rep=rep, fold=fold, seed=seed, model='AE-gate-const', mse=net.loss(Zte, k(len(Zte)))))
    return out


def const_loco(args):
    cfg, path, country, C = args
    c = R.CONFIGS[cfg]
    d, X, Xr, Z, V, parts = R.build(cfg, path)
    geos = d.geo.to_numpy()
    te = geos == country
    tr_c = np.array(sorted(set(geos[~te])))
    kk = int(np.where(np.array(sorted(set(geos))) == country)[0][0])
    k = lambda n: np.full((n, C), 1.0 / C)
    out = []
    for seed in range(R.N_SEED):
        rng = np.random.default_rng(10_000 + 100 * seed + kk)          # identical to run_loco_bootstrap
        va_c = rng.choice(tr_c, size=int(round(0.2 * len(tr_c))), replace=False)
        va = ~te & np.isin(geos, va_c); tr = ~te & ~va
        mu = Z[tr].mean(0); Ztr, Zva, Zte = Z[tr] - mu, Z[va] - mu, Z[te] - mu
        net = G.AE(Z.shape[1], c['H'], c['L'], 'side', s_dim=C, seed=seed)
        net.fit(Ztr, k(len(Ztr)), Zva, k(len(Zva)), seed=seed)
        out.append(dict(country=country, seed=seed, model='AE-gate-const', mse=net.loss(Zte, k(len(Zte)))))
    return out


def corrected(diff, n_tr, n_te):
    J = len(diff); se = np.sqrt((1.0 / J + n_te / n_tr) * diff.var(ddof=1)); q = stats.t.ppf(0.975, J - 1)
    t = diff.mean() / se
    return dict(mean_diff=float(diff.mean()), ci95=[float(diff.mean() - q * se), float(diff.mean() + q * se)],
                p=float(2 * stats.t.sf(abs(t), J - 1)))


def wil(diff):
    w = stats.wilcoxon(diff)
    return dict(median_diff=float(np.median(diff)), W=float(w.statistic), p=float(w.pvalue),
                shapiro_p=float(stats.shapiro(diff).pvalue))


if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    res = {}
    for cfg in ['all20', 'multi8']:
        C = json.load(open(os.path.join(out, f'clustering_{cfg}.json')))['C_star']
        d = G.load_inventory(path, R.CONFIGS[cfg]['gases'])
        with Pool(2) as pool:
            cvc = pd.DataFrame([x for o in pool.map(const_cv, [(cfg, path, r, f, C) for r in range(R.N_REPEAT)
                                                               for f in range(R.N_FOLD)]) for x in o])
            loc = pd.DataFrame([x for o in pool.map(const_loco, [(cfg, path, k, C) for k in sorted(d.geo.unique())])
                                for x in o])
        cvc.to_csv(os.path.join(out, f'ablation_const_cv_raw_{cfg}.csv'), index=False)
        loc.to_csv(os.path.join(out, f'ablation_const_loco_raw_{cfg}.csv'), index=False)
        raw = pd.read_csv(os.path.join(out, f'cv_raw_{cfg}.csv'))
        n_tr = raw.n_tr.mean() + raw.n_va.mean(); n_te = raw.n_te.mean()
        sp = raw.groupby(['model', 'rep', 'fold']).mse.mean().unstack('model')
        sp['AE-gate-const'] = cvc.groupby(['rep', 'fold']).mse.mean()
        lr = pd.read_csv(os.path.join(out, f'loco_raw_{cfg}.csv'))
        per = lr.groupby(['model', 'country']).mse.mean().unstack('model')
        per['AE-gate-const'] = loc.groupby('country').mse.mean()
        rng = np.random.default_rng(2026)
        r = dict(C=C,
                 cv_mean={m: float(sp[m].mean()) for m in ['AE-plain', 'AE-gate-FCM', 'AE-gate-const', 'AE-gate-random']},
                 loco_mean={m: float(per[m].mean()) for m in ['AE-plain', 'AE-gate-FCM', 'AE-gate-const', 'AE-gate-random']},
                 cv_FCM_vs_const=corrected((sp['AE-gate-FCM'] - sp['AE-gate-const']).to_numpy(), n_tr, n_te),
                 cv_const_vs_plain=corrected((sp['AE-gate-const'] - sp['AE-plain']).to_numpy(), n_tr, n_te),
                 loco_FCM_vs_const=LB.boot((per['AE-gate-FCM'] - per['AE-gate-const']).to_numpy(), rng),
                 loco_const_vs_plain=LB.boot((per['AE-gate-const'] - per['AE-plain']).to_numpy(), rng),
                 wilcoxon_cv_FCM_vs_plain=wil((sp['AE-gate-FCM'] - sp['AE-plain']).to_numpy()),
                 wilcoxon_loco_FCM_vs_plain=wil((per['AE-gate-FCM'] - per['AE-plain']).to_numpy()),
                 wilcoxon_loco_FCM_vs_random=wil((per['AE-gate-FCM'] - per['AE-gate-random']).to_numpy()),
                 wilcoxon_loco_FCM_vs_const=wil((per['AE-gate-FCM'] - per['AE-gate-const']).to_numpy()))
        # (c) Ridge-on-PCA-scores vs PCA reconstruction on the full data
        dd, X, Xr, Z, V, parts = R.build(cfg, path)
        Zc = Z - Z.mean(0); L = R.CONFIGS[cfg]['L']
        pca = PCA(L).fit(Zc); S = pca.transform(Zc)
        rec_pca = pca.inverse_transform(S)
        rec_ridge0 = Ridge(alpha=1e-10).fit(S, Zc).predict(S)
        rec_ridge1 = Ridge(alpha=1.0).fit(S, Zc).predict(S)
        r['ridge_vs_pca'] = dict(max_abs_diff_alpha0=float(np.abs(rec_pca - rec_ridge0).max()),
                                 mse_pca=float(((rec_pca - Zc) ** 2).mean()),
                                 mse_ridge_alpha1=float(((rec_ridge1 - Zc) ** 2).mean()))
        res[cfg] = r
    # (d) within-gas FCM, 8-part data
    dd, X, Xr, Z, V, parts = R.build('multi8', path)
    wg = {}
    for g in ['CO2', 'CH4', 'N2O']:
        Zg = Z[dd.gas.to_numpy() == g]
        rows = []
        for C in range(2, 6):
            cen, U, _ = G.fcm(Zg, C, m=2.0, n_init=15, seed=C)
            ind = G.fcm_indices(Zg, U, cen, 2.0)
            rows.append(dict(C=C, SS=ind['SS'], MPC=ind['MPC'], min_cluster_size=int(np.bincount(U.argmax(1)).min()),
                             cluster_sizes=np.bincount(U.argmax(1), minlength=C).tolist()))
        wg[g] = rows
    res['within_gas'] = wg
    json.dump(res, open(os.path.join(out, 'ablation_results.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
