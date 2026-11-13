"""
run_loco_bootstrap.py -- leave-one-country-out (LOCO) validation with country-level bootstrap CIs.

Usage: python run_loco_bootstrap.py <data_csv> <out_dir>

For each of the 31 countries: train on the other 30 (20% of them held out for early stopping),
fit FCM on the training rows only, and record the test MSE of every model on the held-out
country's rows. Paired differences are computed per country (the independent unit) and a
percentile bootstrap over countries (B = 10,000) gives 95% confidence intervals.

Writes: loco_raw_<cfg>.csv, loco_bootstrap_<cfg>.json
"""
import sys, os, json
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghg_core as G
import run_experiments as R

B = 10000
MODELS = ['AE-plain', 'AE-plain-matched', 'AE-gate-FCM', 'AE-gate-random', 'AE-MoE-learned', 'AE-neurofuzzy', 'PCA']


def one_country(args):
    cfg, path, country, C = args
    c = R.CONFIGS[cfg]
    d, X, Xr, Z, V, parts = R.build(cfg, path)
    geos = d.geo.to_numpy()
    te = geos == country
    tr_c = np.array(sorted(set(geos[~te])))
    k = int(np.where(np.array(sorted(set(geos))) == country)[0][0])
    dz, out = Z.shape[1], []
    H_match = G.matched_hidden(dz, c['L'], G.AE(dz, c['H'], c['L'], 'side', s_dim=C).n_params())
    for seed in range(R.N_SEED):
        # deterministic, seed-specific early-stopping split (20% of training countries)
        rng = np.random.default_rng(10_000 + 100 * seed + k)
        va_c = rng.choice(tr_c, size=int(round(0.2 * len(tr_c))), replace=False)
        va = ~te & np.isin(geos, va_c); tr = ~te & ~va
        mu = Z[tr].mean(0); Ztr, Zva, Zte = Z[tr] - mu, Z[va] - mu, Z[te] - mu
        cen, Utr, _ = G.fcm(Ztr, C, m=2.0, n_init=15, seed=7 + seed)
        Uva, Ute = G.fcm_predict(Zva, cen), G.fcm_predict(Zte, cen)
        pr = np.random.default_rng(99 + seed)
        P = tuple(U[pr.permutation(len(U))] for U in (Utr, Uva, Ute))
        specs = {'AE-plain': ('none', (None, None, None), {}),
                 'AE-plain-matched': ('none', (None, None, None), {}),
                 'AE-gate-FCM': ('side', (Utr, Uva, Ute), dict(s_dim=C)),
                 'AE-gate-random': ('side', P, dict(s_dim=C)),
                 'AE-MoE-learned': ('moe', (None, None, None), dict(K=c['K'])),
                 'AE-neurofuzzy': ('nf', (None, None, None), dict(K=c['K']))}
        for name, (gate, S, kw) in specs.items():
            H = H_match if name == 'AE-plain-matched' else c['H']
            net = G.AE(dz, H, c['L'], gate, seed=seed, **kw)
            net.fit(Ztr, S[0], Zva, S[1], seed=seed)
            out.append(dict(country=country, seed=seed, model=name, mse=net.loss(Zte, S[2]), n_te=int(te.sum())))
        pca = PCA(c['L']).fit(np.vstack([Ztr, Zva]))
        out.append(dict(country=country, seed=seed, model='PCA',
                        mse=float(((pca.inverse_transform(pca.transform(Zte)) - Zte) ** 2).mean()), n_te=int(te.sum())))
    return out


def boot(diff, rng):
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    bm = diff[idx].mean(1)
    return dict(mean_diff=float(diff.mean()), ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                p_two_sided=float(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))),
                countries_better=int((diff < 0).sum()), n_countries=int(len(diff)))


def run(cfg, path, out, procs=2):
    C = json.load(open(os.path.join(out, f'clustering_{cfg}.json')))['C_star']
    d = G.load_inventory(path, R.CONFIGS[cfg]['gases'])
    countries = sorted(d.geo.unique())
    with Pool(procs) as pool:
        raw = pd.DataFrame([x for o in pool.map(one_country, [(cfg, path, k, C) for k in countries]) for x in o])
    raw.to_csv(os.path.join(out, f'loco_raw_{cfg}.csv'), index=False)
    per = raw.groupby(['model', 'country']).mse.mean().unstack('model')     # seeds averaged within country
    rng = np.random.default_rng(2026)
    res = dict(cfg=cfg, C=C, n_countries=len(countries), B=B,
               mean_mse={m: float(per[m].mean()) for m in MODELS},
               vs_plain={m: boot((per[m] - per['AE-plain']).to_numpy(), rng) for m in MODELS if m != 'AE-plain'},
               vs_PCA={m: boot((per[m] - per['PCA']).to_numpy(), rng) for m in MODELS if m != 'PCA'},
               gateFCM_vs_random=boot((per['AE-gate-FCM'] - per['AE-gate-random']).to_numpy(), rng),
               gateFCM_vs_matched=boot((per['AE-gate-FCM'] - per['AE-plain-matched']).to_numpy(), rng))
    ps = {m: v['p_two_sided'] for m, v in res['vs_plain'].items()}
    order = sorted(ps, key=ps.get); run = 0
    for k, m in enumerate(order):
        run = max(run, (len(order) - k) * ps[m]); res['vs_plain'][m]['p_holm'] = min(1.0, run)
    json.dump(res, open(os.path.join(out, f'loco_bootstrap_{cfg}.json'), 'w'), indent=1)
    return res


if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    for cfg in ['all20', 'multi8']:
        r = run(cfg, path, out)
        print(cfg, {k: round(v, 3) for k, v in r['mean_mse'].items()})
        for ref in ['vs_plain', 'vs_PCA']:
            for m, v in r[ref].items():
                print(f"  {ref:8s} {m:15s} {v['mean_diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] p={v['p_two_sided']:.3f} better {v['countries_better']}/{v['n_countries']}")
