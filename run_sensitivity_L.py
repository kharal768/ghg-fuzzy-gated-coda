"""Bottleneck-width sensitivity (L = 2, 8) for the all20 configuration; L = 4 is the primary run."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R
path, out = sys.argv[1], sys.argv[2]
C_star = json.load(open(os.path.join(out, 'clustering_all20.json')))['C_star']
for L in (2, 8):
    R.CONFIGS[f'all20_L{L}'] = dict(R.CONFIGS['all20'], L=L)
    summ, tests = R.cv(f'all20_L{L}', path, out, C_star)
    print(L, {k: round(v['mean'], 4) for k, v in sorted(summ.items(), key=lambda kv: kv[1]['mean'])})
    print('   gate-FCM vs plain p =', round(tests['AE-plain']['AE-gate-FCM']['p'], 3))
