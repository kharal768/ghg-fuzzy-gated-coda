"""make_figures.py -- regenerates every figure of the revised manuscript from the frozen outputs.
Usage: python make_figures.py <data_csv> <out_dir> <fig_dir>"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghg_core as G

path, out, fig = sys.argv[1:4]
os.makedirs(fig, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
GAS_LAB = {'CO2': 'CO$_2$', 'CH4': 'CH$_4$', 'N2O': 'N$_2$O', 'HFC_CO2E': 'HFCs', 'PFC_CO2E': 'PFCs',
           'SF6_CO2E': 'SF$_6$', 'NF3_CO2E': 'NF$_3$'}
COL = ['#2a6f97', '#e07a2f', '#3c9d5d', '#c4374a', '#7b5ea7', '#8c6d46', '#d16ba5', '#6b6b6b']

# --- Fig 2: mean closed sector shares per gas (replaces the descriptive maps) -------------------
d = G.load_inventory(path)
P = G.closure(d[G.PARTS_20].to_numpy(float))
M = pd.DataFrame(P, columns=G.PARTS_20).groupby(d.gas.values).mean().reindex(G.GAS_ORDER)
fig2, ax = plt.subplots(figsize=(7.2, 2.9))
im = ax.imshow(M.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
ax.set_yticks(range(7)); ax.set_yticklabels([GAS_LAB[g] for g in M.index])
ax.set_xticks(range(20)); ax.set_xticklabels([p.split('_')[1] for p in G.PARTS_20], rotation=90)
for i in range(7):
    for j in range(20):
        v = M.values[i, j]
        if v >= 0.05:
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5, color='white' if v > 0.55 else 'black')
        elif v == 0:
            ax.text(j, i, '0', ha='center', va='center', fontsize=6, color='#bbbbbb')
ax.set_xlabel('CRF part (Table 1)'); fig2.colorbar(im, ax=ax, fraction=0.025, label='mean share')
fig2.tight_layout(); fig2.savefig(os.path.join(fig, 'fig_shares.pdf')); plt.close(fig2)

# --- Fig 3: FCM validity indices ----------------------------------------------------------------
fig3, axs = plt.subplots(1, 4, figsize=(7.4, 2.1))
for cfg, col, lab in [('all20', COL[0], '7 gases, D=20'), ('multi8', COL[1], 'CO$_2$/CH$_4$/N$_2$O, D=8')]:
    t = pd.DataFrame(json.load(open(os.path.join(out, f'clustering_{cfg}.json')))['indices'])
    for a, k, yl in zip(axs, ['PC', 'MPC', 'SS', 'ARI_gas'], ['PC', 'MPC', 'Silhouette', 'ARI vs gas']):
        a.plot(t.C, t[k], 'o-', ms=3, color=col, label=lab); a.set_xlabel('C'); a.set_title(yl, fontsize=9)
for a in axs:
    a.set_xticks(range(2, 11, 2))
axs[0].plot(range(2, 11), [1 / c for c in range(2, 11)], 'k:', lw=1, label='1/C (lower bound)')
axs[0].legend(fontsize=6.5, frameon=False)
fig3.tight_layout(); fig3.savefig(os.path.join(fig, 'fig_validity.pdf')); plt.close(fig3)

# --- Fig 4: FCM partitions in the first two principal ILR directions --------------------------
fig4, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
mk = {'CO2': 'o', 'CH4': 's', 'N2O': '^', 'HFC_CO2E': 'D', 'PFC_CO2E': 'v', 'SF6_CO2E': 'P', 'NF3_CO2E': 'X'}
for a, cfg, ttl in zip(axs, ['all20', 'multi8'], ['(a) 7 gases, 20 parts', '(b) CO$_2$, CH$_4$, N$_2$O, 8 parts']):
    f = np.load(os.path.join(out, f'fcm_full_{cfg}.npz'), allow_pickle=True)
    Z, U, gas = f['Z'], f['U'], f['gas']
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    S = Zc @ Vt[:2].T
    lab = U.argmax(1)
    for g in [x for x in G.GAS_ORDER if x in set(gas)]:
        m = gas == g
        a.scatter(S[m, 0], S[m, 1], c=[COL[k % 8] for k in lab[m]], marker=mk[g], s=18,
                  edgecolor='k', linewidth=0.3, alpha=0.4 + 0.6 * U.max(1)[m].mean(), label=GAS_LAB[g])
    a.set_title(ttl, fontsize=9); a.set_xlabel('PC1 of ILR'); a.set_ylabel('PC2 of ILR')
    leg = a.legend(loc='center', fontsize=6.5, frameon=False, markerscale=0.9, title='marker = gas; colour = FCM cluster',
                   title_fontsize=6.5)
    for h in leg.legend_handles:
        h.set_facecolor('white')
fig4.tight_layout(); fig4.savefig(os.path.join(fig, 'fig_clusters.png'), dpi=300); plt.close(fig4)

# --- Fig 5: grouped-CV results -------------------------------------------------------------------
ORDER = ['PCA', 'RF-on-PCA', 'AE-plain', 'AE-plain-matched', 'AE-concat', 'AE-gate-FCM', 'AE-gate-hard',
         'AE-gate-random', 'AE-gate-gaslabel', 'AE-MoE-learned', 'AE-neurofuzzy']
fig5, axs = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=False)
for a, cfg, ttl in zip(axs, ['all20_L2', 'all20', 'all20_L8'],
                       ['7 gases, L=2', '7 gases, L=4 (primary)', '7 gases, L=8']):
    raw = pd.read_csv(os.path.join(out, f'cv_raw_{cfg}.csv'))
    sp = raw.groupby(['model', 'rep', 'fold']).mse.mean().unstack('model')
    data = [sp[m].values for m in ORDER]
    bp = a.boxplot(data, orientation='horizontal', widths=0.6, patch_artist=True, showfliers=False)
    for i, b in enumerate(bp['boxes']):
        b.set_facecolor(COL[1] if 'gate' in ORDER[i] or ORDER[i] in ('AE-MoE-learned', 'AE-neurofuzzy') else
                        (COL[0] if ORDER[i].startswith('AE') else '#bbbbbb'))
    a.set_yticks(range(1, len(ORDER) + 1)); a.set_yticklabels(ORDER if a is axs[0] else [], fontsize=7)
    a.set_title(ttl, fontsize=9); a.set_xlabel('test MSE (ILR)')
fig5.tight_layout(); fig5.savefig(os.path.join(fig, 'fig_cv.pdf')); plt.close(fig5)

fig6, a = plt.subplots(figsize=(3.6, 3.2))
raw = pd.read_csv(os.path.join(out, 'cv_raw_multi8.csv'))
sp = raw.groupby(['model', 'rep', 'fold']).mse.mean().unstack('model')
bp = a.boxplot([sp[m].values for m in ORDER], orientation='horizontal', widths=0.6, patch_artist=True, showfliers=False)
for i, b in enumerate(bp['boxes']):
    b.set_facecolor(COL[1] if 'gate' in ORDER[i] or ORDER[i] in ('AE-MoE-learned', 'AE-neurofuzzy') else
                    (COL[0] if ORDER[i].startswith('AE') else '#bbbbbb'))
a.set_yticks(range(1, len(ORDER) + 1)); a.set_yticklabels(ORDER, fontsize=7)
a.set_title('CO$_2$/CH$_4$/N$_2$O, D=8, L=2', fontsize=9); a.set_xlabel('test MSE (ILR)')
fig6.tight_layout(); fig6.savefig(os.path.join(fig, 'fig_cv_multi8.pdf')); plt.close(fig6)

# --- Fig 7: learned gate profiles (cluster x hidden unit) ---------------------------------------
fig7, axs = plt.subplots(2, 1, figsize=(7.2, 3.4), gridspec_kw=dict(hspace=0.6))
for a, cfg in zip(axs, ['all20', 'multi8']):
    Pp = np.load(os.path.join(out, f'gate_profile_example_{cfg}.npy'))
    im = a.imshow(Pp, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    a.set_ylabel('FCM cluster'); a.set_yticks(range(Pp.shape[0]))
    a.set_title({'all20': '7 gases (H=64)', 'multi8': 'CO$_2$/CH$_4$/N$_2$O (H=32)'}[cfg], fontsize=8.5)
axs[1].set_xlabel('hidden unit'); fig7.colorbar(im, ax=axs, fraction=0.02, label='mean gate value')
fig7.savefig(os.path.join(fig, 'fig_gates.pdf'), bbox_inches='tight'); plt.close(fig7)
print('figures written to', fig)
